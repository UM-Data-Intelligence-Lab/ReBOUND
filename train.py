#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def run(args):
    from ReBOUND.training.assembly import initialize_bundle, assemble_experts
    from ReBOUND.training.evaluation import evaluate
    from ReBOUND.training.residual import fit_shared, fit_scope
    from ReBOUND.training.routing import fit_routing
    from ReBOUND.training.runner import train_experts, training_signature
    from ReBOUND.utils.io import read_json, sha256, write_json
    output = Path(args.output).expanduser().resolve() if args.output else ROOT.parent / "ReBOUND_runs" / time.strftime("%Y%m%d_%H%M%S")
    if output.is_relative_to(ROOT):
        raise ValueError("Choose --output outside the release directory.")
    if output.exists() and not args.resume:
        raise ValueError("Output exists; choose a fresh directory or use --resume.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / "Data/training/manifest.json")
    for relative, record in manifest["files"].items():
        path = ROOT / relative
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Training input mismatch: {relative}")
    signature = training_signature(ROOT)
    snapshot = output / "training_source.json"
    if snapshot.exists() and read_json(snapshot) != signature:
        raise ValueError("Training code or configuration changed; use a fresh run directory.")
    write_json(snapshot, signature)
    started = time.monotonic()
    neural_count = train_experts(ROOT, output, args.gpus.split(","), args.jobs_per_gpu,
                                args.gpu_memory_gib, resume=args.resume)
    if args.stage == "pretrain":
        print(f"Expert pretraining complete: {output}")
        return
    weights = output / "weights"
    if not weights.exists():
        initialize_bundle(ROOT, weights)
        assemble_experts(ROOT, output / "experts/target", weights)
        if args.foundation_dir:
            from ReBOUND.training.foundation import assemble_foundation
            assemble_foundation(args.foundation_dir, weights)
        fit_routing(ROOT, weights)
        fit_shared(ROOT, weights)
    scopes = [scope for scope in range(24) if not (weights / "residual" / f"scope_{scope:02d}.joblib").exists()]
    with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
        futures = [pool.submit(fit_scope, ROOT, weights, scope) for scope in scopes]
        for future in futures:
            future.result()
    if not (output / "evaluation").exists():
        metrics = evaluate(ROOT, weights, output / "evaluation")
    else:
        metrics = read_json(output / "evaluation/metrics.json")
    write_json(output / "training.json", dict(status="complete", neural_models=neural_count, residual_models=799,
        routing_scopes=24, seconds=time.monotonic() - started, metrics=metrics,
        foundation_adapters=4 if args.foundation_dir else 0,
        inputs="prepared masked expert inputs, pretrained foundation/spatial features, cross-fitted residual features"))
    print(f"Training and evaluation complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "ReBOUND/config/rebound.json"))
    parser.add_argument("--reproduction", choices=["true", "false"], default="true")
    parser.add_argument("--output")
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU indices or UUIDs")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--gpu-memory-gib", type=float, default=2.)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--stage", choices=["all", "pretrain"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--foundation-dir", type=Path, help="Output of pretrain.py --foundation")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reproduction == "true":
        import subprocess
        command = [sys.executable, "-B", str(ROOT / "reproduce.py"), "--config", args.config]
        if args.output:
            command += ["--output", args.output]
        raise SystemExit(subprocess.call(command))
    from reproduce import environment_check
    from ReBOUND.utils.io import read_json
    from importlib.metadata import version
    environment_check()
    if version("torch").split("+")[0] != "2.3.1":
        raise RuntimeError("Use PyTorch 2.3.1: python -m pip install -r requirements.txt")
    if read_json(args.config)["framework"] != "ReBOUND":
        raise ValueError("Expected a ReBOUND configuration.")
    if args.cpu_workers not in range(1, 17) or args.jobs_per_gpu not in range(1, 9):
        raise ValueError("Use 1–16 CPU workers and 1–8 jobs per GPU.")
    if not 0 < args.gpu_memory_gib < 1024:
        raise ValueError("GPU memory budget must be positive and finite.")
    if args.worker:
        run(args)
    else:
        from ReBOUND.utils.runtime import supervise
        budget = max(3 * args.cpu_workers, 6 * len(args.gpus.split(",")) * args.jobs_per_gpu) + 3
        raise SystemExit(supervise([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                                   memory_gib=budget, virtual_limit=False))


if __name__ == "__main__":
    main()
