from pathlib import Path
import shutil

import numpy as np

from ..utils.io import load_arrays, read_json


def initialize_bundle(root, output):
    root, output = Path(root), Path(output)
    for directory in ("experts", "routing"):
        (output / directory).mkdir(parents=True, exist_ok=False)
    for name in ("diagonal", "off_diagonal", "residual_features", "spatial"):
        shutil.copyfile(root / "Pretrained/experts" / (name + ".npz"), output / "experts" / (name + ".npz"))
    for name in ("diagonal", "off_diagonal", "spatial"):
        shutil.copyfile(root / "Pretrained/routing" / (name + ".json"), output / "routing" / (name + ".json"))


def assemble_experts(root, runs, output):
    root, runs, output = Path(root), Path(runs), Path(output)
    config = read_json(root / "ReBOUND/config/periodicity.json")
    means = {}
    for recipe, component, epoch in [("conservative", "diag", 70), ("wide", "off", 80), ("regularized", "off", 80)]:
        values = []
        for seed in config["seeds"]:
            values.append(load_arrays(runs / "periodicity" / recipe / component / str(seed) / "predictions.npz")["predictions"][epoch - 1].astype(float))
        means[recipe] = np.mean(values, axis=0)
    for component, recipe in [("diagonal", "conservative"), ("off_diagonal", "wide")]:
        data = load_arrays(output / "experts" / (component + ".npz"))
        data["raw"][:, 0] = means[recipe]
        np.savez_compressed(output / "experts" / (component + ".npz"), **data)
    data = load_arrays(output / "experts/residual_features.npz")
    data["predictions"][0] = means["regularized"]
    np.savez_compressed(output / "experts/residual_features.npz", **data)
    config = read_json(root / "ReBOUND/config/grouped.json")
    result = {}
    for index, group in enumerate(["middle", "rare", "top"]):
        means = {}
        for loss in config["groups"][group]["training"]:
            values = []
            for seed in config["seeds"]:
                data = load_arrays(runs / "grouped" / group / loss / str(seed) / "predictions.npz")
                values.append(data["flow_coefficient"].astype(float))
            means[loss] = np.mean(values, axis=0)
        coefficients = .5 * means["tweedie"] + .5 * means["mse"] if group == "top" else next(iter(means.values()))
        result[f"ratios_{index}"] = np.clip((coefficients + .02) / (data["occurrence_prior"][None] + .02), .5, 2)
        result[f"positions_{index}"] = data["selected_edges"]
        result[f"checkpoints_{index}"] = data["checkpoints"]
    np.savez_compressed(output / "experts/grouped.npz", **result)
