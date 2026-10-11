from pathlib import Path
import subprocess
import sys

import numpy as np

from ..model.rebound import ReBOUND
from ..utils.evaluation import export_submission, score_validation, verify_roundtrip
from ..utils.io import read_json, write_json


def evaluate(root, weights, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = ReBOUND(root, weights)
    prediction = model.predict(23)
    np.savez_compressed(output / "prediction.npz", **prediction)
    values = export_submission(output / "submission.tsv", prediction)
    if verify_roundtrip(output / "submission.tsv", prediction) != values or values != 82708:
        raise ValueError("Unexpected submission support.")
    result = subprocess.run([sys.executable, str(root / "ReBOUND/utils/official_validator.py"),
                             str(output / "submission.tsv")], capture_output=True, text=True, timeout=60)
    (output / "validator.log").write_text(result.stdout + result.stderr)
    if result.returncode or result.stdout.strip() != "Validation passed!":
        raise ValueError("Submission failed the official format validator.")
    diagonal, off = [], []
    for scope in range(23):
        value = model.predict(scope)
        position = model.observations["positions"][scope]
        diagonal.append(value["candidate_diag"][position])
        off.append(value["candidate_off"][position])
        print(f"Evaluated held date {scope + 1}/23", flush=True)
    validation = dict(prediction_dates=model.observations["dates"], active_nodes=prediction["active_nodes"],
                      edges=prediction["edges"], candidate_diag=np.stack(diagonal), candidate_off=np.stack(off))
    np.savez_compressed(output / "validation.npz", **validation)
    metrics = score_validation(validation, model.observations, output / "daily_metrics.csv")
    write_json(output / "metrics.json", metrics)
    expected = read_json(root / "ReBOUND/config/expected.json")["metrics"]["Val23"]
    difference = {key: metrics["Val23"][key] - expected[key] for key in expected}
    write_json(output / "comparison.json", dict(reference=expected, retrained=metrics["Val23"], difference=difference))
    print(f"Retrained Val23: {metrics['Val23']}", flush=True)
    return metrics
