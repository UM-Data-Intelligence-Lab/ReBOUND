import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def memory_state():
    state = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        state[key] = int(value.split()[0]) * 1024
    return state


def cgroup_problem(reserve=0):
    text = Path("/proc/self/cgroup").read_text()
    entry = next((line[3:] for line in text.splitlines() if line.startswith("0::")), None)
    if entry is None:
        return None
    root = Path("/sys/fs/cgroup")
    current = root / entry.lstrip("/")
    while current != root:
        for kind in ("memory", "memory.swap"):
            maximum = current / (kind + ".max")
            if maximum.exists():
                limit_text = maximum.read_text().strip()
                used = int((current / (kind + ".current")).read_text())
                if limit_text != "max":
                    limit = int(limit_text)
                    extra = reserve if kind == "memory" else 0
                    if limit > 0 and used + extra >= .9 * limit:
                        return f"{kind} headroom below 10% in {current}"
        current = current.parent
    return None


def resource_problem(reserve=0):
    state = memory_state()
    if state["MemAvailable"] - reserve <= .1 * state["MemTotal"]:
        return "Host RAM headroom would fall to 10% or less."
    if os.environ.get("REBOUND_IGNORE_HOST_SWAP") != "1" and state["SwapTotal"] and state["SwapFree"] <= .1 * state["SwapTotal"]:
        return "Host swap usage reached 90%."
    return cgroup_problem(reserve)


def paging_state():
    counts = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    pressure_file = Path("/proc/pressure/memory")
    pressure = 0.
    if pressure_file.exists():
        line = pressure_file.read_text().splitlines()[0]
        pressure = float(dict(item.split("=") for item in line.split()[1:])["avg10"])
    return int(counts["pswpin"]) + int(counts["pswpout"]), pressure


def terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def process_group_rss(group):
    total = 0
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                total += int((directory / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def supervise(command, memory_gib=3, virtual_limit=True):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The verified release entry point requires Linux.")
    maximum = int(memory_gib * 1024**3)
    problem = resource_problem(maximum)
    if problem:
        raise RuntimeError("Resource preflight stopped: " + problem)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

    def bound_address_space():
        resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))

    process = subprocess.Popen(command, env=environment, start_new_session=True,
                               preexec_fn=bound_address_space if virtual_limit else None)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
    previous, _ = paging_state()
    pressure_samples = 0
    try:
        while process.poll() is None:
            time.sleep(2)
            problem = resource_problem()
            if not virtual_limit and process_group_rss(process.pid) > maximum:
                problem = f"Process-group RSS exceeded {memory_gib:g} GiB."
            paging, pressure = paging_state()
            pressure_samples = pressure_samples + 1 if paging > previous and pressure >= 1. else 0
            previous = paging
            if pressure_samples >= 3:
                problem = "Sustained swapping accompanied by memory pressure."
            if problem:
                terminate(process)
                raise RuntimeError("Resource monitor stopped this run: " + problem)
    except BaseException:
        terminate(process)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return process.returncode
