import os
from pathlib import Path
import subprocess
import sys
import time

from ..utils.io import read_json, sha256, write_json
from ..utils.runtime import resource_problem


def gpu_state():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    state = {}
    for line in result.stdout.splitlines():
        index, uuid, free, utilization = [x.strip() for x in line.split(",")]
        state[index] = state[uuid] = dict(uuid=uuid, free_mib=int(free), utilization=int(utilization))
    return state


def expert_jobs(root, output, task="target"):
    root, output = Path(root), Path(output)
    jobs = []
    for recipe, component in [("conservative", "diag"), ("wide", "off"), ("regularized", "off")]:
        for seed in read_json(root / "ReBOUND/config/periodicity.json")["seeds"]:
            directory = output / "experts" / task / "periodicity" / recipe / component / str(seed)
            jobs.append((directory, ["periodicity", "--recipe", recipe, "--component", component, "--seed", str(seed)]))
    config = read_json(root / "ReBOUND/config/grouped.json")
    for group, options in config["groups"].items():
        for loss in options["training"]:
            for seed in config["seeds"]:
                directory = output / "experts" / task / "grouped" / group / loss / str(seed)
                jobs.append((directory, ["grouped", "--group", group, "--loss", loss, "--seed", str(seed)]))
    return [(directory, arguments + ["--task", task]) for directory, arguments in jobs]


def train_experts(root, output, devices, jobs_per_gpu=1, gpu_memory_gib=2, task="target", resume=False):
    root, output = Path(root), Path(output)
    (output / "logs").mkdir(exist_ok=True)
    jobs = expert_jobs(root, output, task)
    pending = []
    for directory, arguments in jobs:
        if (directory / "training.json").exists() and resume:
            record = read_json(directory / "training.json")
            if not record["from_random_initialization"] or not record["weights_changed"]:
                raise ValueError(f"Incomplete training record: {directory}")
            continue
        if directory.exists():
            raise ValueError(f"Unfinished expert output exists: {directory}. Move it aside before retrying.")
        pending.append((directory, arguments))
    devices = list(dict.fromkeys(devices))
    if len(devices) > 6 or jobs_per_gpu < 1:
        raise ValueError("Use one to six GPUs and a positive jobs-per-GPU value.")
    state = gpu_state()
    devices = [state[device]["uuid"] for device in devices]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != "-1":
        allowed = {state[device.strip()]["uuid"] for device in visible.split(",")}
        if not set(devices) <= allowed:
            raise ValueError("Requested GPUs exceed CUDA_VISIBLE_DEVICES.")
    if visible == "-1":
        raise ValueError("CUDA_VISIBLE_DEVICES disables GPU use.")
    running, last_progress = [], time.monotonic()
    telemetry = output / "logs/gpu.jsonl"
    try:
        while pending or running:
            state = gpu_state()
            for entry in running[:]:
                process, stream, gpu, directory = entry
                if process.poll() is not None:
                    stream.close()
                    running.remove(entry)
                    if process.returncode:
                        raise RuntimeError(f"Expert training failed: {directory}; see {stream.name}")
                    last_progress = time.monotonic()
                    print(f"Completed: {directory.relative_to(output)}", flush=True)
            for gpu in devices:
                count = sum(entry[2] == gpu for entry in running)
                required = int(gpu_memory_gib * 1024) + 768
                if not pending or count >= jobs_per_gpu or state[gpu]["free_mib"] < required:
                    continue
                # Reserve remaining growth of workers that have not produced a profile yet.
                growth = sum(required for entry in running if entry[2] == gpu and not (entry[3] / "profile.json").exists())
                if state[gpu]["free_mib"] - growth < required:
                    continue
                problem = resource_problem((len(running) + 1) * 6 * 1024**3)
                if problem:
                    raise RuntimeError("Training resource preflight: " + problem)
                directory, arguments = pending.pop(0)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
                logfile = output / "logs" / ("_".join(directory.relative_to(output / "experts").parts) + ".log")
                stream = logfile.open("w")
                command = [sys.executable, "-B", "-m", "ReBOUND.training.worker", *arguments,
                           "--root", str(root), "--output", str(directory), "--device", "cuda:0",
                           "--gpu-memory-gib", str(gpu_memory_gib)]
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                running.append((process, stream, gpu, directory))
                print(f"Started: {directory.relative_to(output)} on {gpu}", flush=True)
                last_progress = time.monotonic()
            with telemetry.open("a") as stream:
                import json
                stream.write(json.dumps(dict(time=time.time(), active=len(running),
                    devices={gpu: state[gpu] for gpu in devices})) + "\n")
            if pending and not running and time.monotonic() - last_progress > 60:
                raise RuntimeError("Selected GPUs do not have the requested memory headroom.")
            if pending or running:
                time.sleep(2)
    finally:
        for process, stream, _, _ in running:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 12
        for process, stream, _, _ in running:
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
    return len(jobs)


def training_signature(root):
    root = Path(root)
    paths = list((root / "ReBOUND/training").glob("*.py"))
    paths += [root / "ReBOUND/config/periodicity.json", root / "ReBOUND/config/grouped.json",
              root / "Data/training/manifest.json"]
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}
