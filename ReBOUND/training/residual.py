from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..model.rebound import ReBOUND
from ..model.modules.residual import ProgressiveResidualLearning
from ..utils.io import load_arrays, read_json, write_json


def fit_tree(data, parameters):
    weights = data.get("w")
    if weights is not None and weights.size == 0:
        weights = None
    return HistGradientBoostingRegressor(**parameters).fit(data["x"], data["y"], sample_weight=weights)


def fit_shared(root, output):
    source = Path(root) / "Data/training/residual"
    directory = Path(output) / "residual"
    directory.mkdir(parents=True, exist_ok=False)
    parameters = read_json(source / "parameters.json")
    models = {}
    for name in ("low_flow", "expert", "boundary", "global", "pooled", "multiwindow", "teacher"):
        data = load_arrays(source / "shared" / (name + ".npz"))
        models[name] = fit_tree(data, parameters[name])
        print(f"Fitted residual head: {name} ({len(data['y'])} rows)", flush=True)
    joblib.dump(models, directory / "shared.joblib", compress=3)
    data = load_arrays(source / "modes.npz")
    residual = data["fit_residual"]
    _, singular, right = np.linalg.svd(residual, full_matrices=False)
    basis = right[:8].T.copy()
    first = basis[:, :4].copy()
    mean = (residual @ first).mean(axis=0)
    random = np.random.default_rng(20260923).normal(size=first.shape)
    random -= first @ (first.T @ random)
    state = load_arrays(source / "metadata.npz")
    state.update(U4=first, U8=basis, delta=.1 * (np.tile(mean, (86, 1)) @ first.T),
                 random_complement=np.linalg.qr(random)[0],
                 original_random=np.linalg.qr(np.random.default_rng(20260920).normal(size=first.shape))[0])
    np.savez_compressed(directory / "basis.npz", **state)
    write_json(directory / "basis_fit.json", dict(rows=len(residual), singular_values=singular[:8].tolist()))
    write_json(directory / "recipe.json", read_json(Path(root) / "Pretrained/residual/recipe.json"))


def fit_scope(root, output, scope):
    started = time.monotonic()
    root, output = Path(root), Path(output)
    source = root / "Data/training/residual"
    parameters = read_json(source / "parameters.json")
    base_model = ReBOUND(root, output, residual=False)
    state = load_arrays(output / "residual/basis.npz")
    residual = ProgressiveResidualLearning(output / "residual", state,
        read_json(output / "residual/recipe.json"), base_model.observations,
        load_arrays(output / "experts/residual_features.npz")["predictions"])
    _, base = base_model.base_prediction(scope)
    _, prediction = residual.prefix(base, scope)
    held = () if scope == 23 else (scope,)
    fit = np.flatnonzero(np.arange(23) != scope)
    features = residual._features(prediction, held, 14)
    x = features[state["positions"][fit]].reshape(-1, 14).astype(float)
    y = (residual.truth[fit] - prediction[state["positions"][fit]]).ravel()
    models = {"boundary": fit_tree(dict(x=x, y=y), parameters["conditional_boundary"])}
    for stage in ("residual_ensemble", "temporal_ensemble"):
        data = load_arrays(source / f"scope_{scope:02d}" / (stage + ".npz"))
        np.testing.assert_array_equal(data["fit"], fit)
        members = []
        for member in range(16):
            rng = np.random.default_rng(np.random.SeedSequence([20260925, scope, member]))
            left, right = np.flatnonzero(fit < 9), np.flatnonzero(fit >= 9)
            drawn = np.sort(np.r_[rng.choice(left, len(left), replace=True),
                                  rng.choice(right, len(right), replace=True)])
            indices = (drawn[:, None] * 509 + np.arange(509)).ravel()
            weights = data["w"][indices].copy()
            weights *= 42756 / weights.sum()
            members.append(fit_tree(dict(x=data["x"][indices], y=data["y"][indices], w=weights), parameters["ensemble"]))
        models[stage] = members
    joblib.dump(models, output / "residual" / f"scope_{scope:02d}.joblib", compress=3)
    write_json(output / "residual" / f"scope_{scope:02d}.json",
               dict(scope=scope, fitted_trees=33, fit_dates=base_model.observations["dates"][fit].tolist(),
                    seconds=time.monotonic() - started))
    print(f"Fitted residual scope {scope + 1}/24", flush=True)
