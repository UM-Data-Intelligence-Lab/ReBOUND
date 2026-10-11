#!/usr/bin/env python3
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def environment_check():
    required = {"numpy": "1.26.4", "scipy": "1.13.1", "scikit-learn": "1.7.2", "joblib": "1.6.0", "threadpoolctl": "3.6.0", "cloudpickle": "3.1.2"}
    if sys.version_info[:2] not in [(3, 10), (3, 11)]:
        raise RuntimeError("Use Python 3.10 or 3.11 for this release.")
    for package, expected in required.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            raise RuntimeError(f"{package} {actual}; need {expected}. Run: python -m pip install -r requirements.txt")
    return required


def run(args):
    import numpy as np
    from ReBOUND.model.rebound import ReBOUND
    from ReBOUND.utils.evaluation import export_submission, score_validation, verify_roundtrip
    from ReBOUND.utils.io import read_json, sha256, verify_arrays, verify_bundle, write_json

    started = time.monotonic()
    manifest = verify_bundle(ROOT)
    expected = read_json(ROOT / "ReBOUND/config/expected.json")
    if args.check:
        print(f"Verified {len(manifest['files'])} release artifacts; environment is ready.")
        return
    config = read_json(args.config)
    output = Path(args.output).expanduser().resolve() if args.output else (ROOT / config["output_directory"]).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Choose --output outside the release directory.")
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}. Choose a fresh directory with --output.")
    output.mkdir(parents=True)
    model = ReBOUND(ROOT)
    print("Generating ReBOUND deployment predictions...", flush=True)
    prediction = model.predict(23)
    verify_arrays(prediction, expected["prediction"])
    np.savez_compressed(output / "prediction.npz", **prediction)
    values = export_submission(output / "submission.tsv", prediction)
    if verify_roundtrip(output / "submission.tsv", prediction) != expected["values"] or values != expected["values"]:
        raise ValueError("Submission support differs from the reference result.")
    digest = sha256(output / "submission.tsv")
    if digest != expected["submission_sha256"]:
        raise ValueError("Submission differs from the reference result.")
    validator = subprocess.run([sys.executable, str(ROOT / "ReBOUND/utils/official_validator.py"),
                                str(output / "submission.tsv")], text=True, capture_output=True, timeout=60)
    (output / "validator.log").write_text(validator.stdout + validator.stderr, encoding="utf-8")
    if validator.returncode != 0 or validator.stdout.strip() != "Validation passed!":
        raise ValueError("The official HuMob format validator did not pass.")
    metrics = None
    if not args.submission_only:
        diag, off = [], []
        for scope, day in enumerate(model.observations["dates"]):
            result = model.predict(scope)
            position = model.observations["positions"][scope]
            diag.append(result["candidate_diag"][position])
            off.append(result["candidate_off"][position])
            print(f"Val23: {scope + 1}/23 ({int(day)})", flush=True)
        validation = dict(prediction_dates=model.observations["dates"], active_nodes=prediction["active_nodes"],
                          edges=prediction["edges"], candidate_diag=np.stack(diag), candidate_off=np.stack(off))
        verify_arrays(validation, expected["validation"])
        np.savez_compressed(output / "validation.npz", **validation)
        metrics = score_validation(validation, model.observations, output / "daily_metrics.csv")
        for group, components in expected["metrics"].items():
            for component, value in components.items():
                if abs(metrics[group][component] - value) > 1e-12:
                    raise ValueError(f"Metric mismatch: {group}/{component}")
        write_json(output / "metrics.json", metrics)
    report = dict(status="passed", source=expected["source"], mode="frozen_experts_and_fitted_residual_models",
                  deployment_arrays_exact=True, validation_arrays_exact=not args.submission_only,
                  submission_byte_exact=True, submission_sha256=digest, target_days=58, values=values,
                  official_validator_passed=True, metrics=metrics, elapsed_seconds=time.monotonic() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  environment=environment_check(), bundle_sha256=sha256(ROOT / "Pretrained/manifest.json"))
    write_json(output / "verification.json", report)
    if metrics:
        scores = metrics["Val23"]
        print(f"Val23: Diag={scores['diag']:.10f} Off={scores['offdiag']:.10f} Combined={scores['combined']:.10f}")
    print(f"PASS: 58 days, {values:,} values; ReBOUND submission verified.\nOutput: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "ReBOUND/config/rebound.json"), help="ReBOUND release configuration")
    parser.add_argument("--output", help="Fresh output directory outside the repository")
    parser.add_argument("--submission-only", action="store_true", help="Generate and verify the final submission without rescoring Val23")
    parser.add_argument("--check", action="store_true", help="Check dependencies and all artifact hashes without inference")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        environment_check()
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if config["framework"] != "ReBOUND" or config["inference"] != "frozen_experts_and_fitted_residual_models":
            raise ValueError("Use the released ReBOUND inference configuration.")
        if config["data_directory"] != "Data" or config["pretrained_directory"] != "Pretrained":
            raise ValueError("Keep the released Data and Pretrained directories beside reproduce.py.")
        if args.worker:
            run(args)
        else:
            from ReBOUND.utils.runtime import supervise
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], config["memory_gib"]))
    except (RuntimeError, ValueError, FileNotFoundError, MemoryError) as error:
        print(f"Reproduction stopped: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
