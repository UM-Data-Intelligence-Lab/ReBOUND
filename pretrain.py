#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def foundation(args):
    from ReBOUND.training.runner import gpu_state
    from ReBOUND.utils.runtime import resource_problem
    output = Path(args.output).expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Choose --output outside the release directory.")
    output.mkdir(parents=True, exist_ok=False)
    state = gpu_state()
    devices = list(dict.fromkeys(state[x]["uuid"] for x in args.gpus.split(",")))
    if len(devices) > 6:
        raise ValueError("At most six GPUs may be used.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and (visible == "-1" or not set(devices) <= {state[x]["uuid"] for x in visible.split(",")}):
        raise ValueError("Requested GPUs exceed CUDA_VISIBLE_DEVICES.")
    pending, running = list(range(4)), []
    progress = time.monotonic()
    try:
        while pending or running:
            state = gpu_state()
            for item in running[:]:
                process, stream, gpu, group = item
                if process.poll() is not None:
                    stream.close()
                    running.remove(item)
                    if process.returncode:
                        raise RuntimeError(f"Foundation group {group} failed; see {output}/group_{group}.log")
                    print(f"Completed foundation group {group}.", flush=True)
                    progress = time.monotonic()
            for gpu in devices:
                if not pending or any(item[2] == gpu for item in running):
                    continue
                if state[gpu]["free_mib"] < args.gpu_memory_gib * 1024 + 1024:
                    continue
                problem = resource_problem((len(running) + 1) * 8 * 1024**3)
                if problem:
                    raise RuntimeError(problem)
                group = pending.pop(0)
                command = [sys.executable, "-B", "-m", "ReBOUND.training.worker", "foundation",
                    "--root", str(ROOT), "--output", str(output / f"group_{group}"), "--foundation-group", str(group),
                    "--device", "cuda:0", "--gpu-memory-gib", str(args.gpu_memory_gib)]
                if args.model_path:
                    command += ["--model-path", args.model_path]
                stream = (output / f"group_{group}.log").open("w")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                                   NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1")
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                running.append((process, stream, gpu, group))
                print(f"Started foundation group {group} on {gpu}.", flush=True)
                progress = time.monotonic()
            with (output / "gpu.jsonl").open("a") as stream:
                import json
                stream.write(json.dumps(dict(time=time.time(), devices={gpu: state[gpu] for gpu in devices})) + "\n")
            if pending and not running and time.monotonic() - progress > 60:
                raise RuntimeError("Selected GPUs lack foundation-training memory headroom.")
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


def main():
    if "--foundation" not in sys.argv:
        from train import main as train
        sys.argv.extend(["--reproduction", "false", "--stage", "pretrain"])
        return train()
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--gpu-memory-gib", type=float, default=5.)
    parser.add_argument("--model-path", help="Optional local Chronos-2 base-model directory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        foundation(args)
    else:
        from ReBOUND.utils.runtime import supervise
        raise SystemExit(supervise([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                                   8 * len(args.gpus.split(",")) + 2, virtual_limit=False))


if __name__ == "__main__":
    main()
