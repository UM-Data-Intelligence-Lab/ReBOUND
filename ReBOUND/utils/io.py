import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_arrays(path, allow_missing=()):
    # Never materialize a dense T x N x N OD tensor.
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024**2:
            raise ValueError(f"Array archive exceeds the 256 MiB loading budget: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    for key, value in result.items():
        if value.dtype.kind in "fc" and (np.isinf(value).any() or (key not in allow_missing and np.isnan(value).any())):
            raise ValueError(f"Non-finite array: {path}:{key}")
    return result


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = read_json(root / "Pretrained/manifest.json")
    for relative, record in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or external release artifact: {relative}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    return manifest


def verify_arrays(arrays, expected):
    if set(arrays) != set(expected):
        raise ValueError("Prediction schema differs from the reference result.")
    for key, value in arrays.items():
        record = expected[key]
        if list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]:
            raise ValueError(f"Prediction shape or dtype differs: {key}")
        if hashlib.sha256(value.tobytes(order="C")).hexdigest() != record["sha256"]:
            raise ValueError(f"Prediction values differ from the reference result: {key}")
