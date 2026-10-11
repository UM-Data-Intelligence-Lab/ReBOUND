from pathlib import Path

import numpy as np

from ..utils.io import load_arrays, read_json
from .simplex import endpoint_simplex, simplex_quadratic


def fit_routing(root, output):
    root, output = Path(root), Path(output)
    observations = load_arrays(root / "Data/observations.npz")
    statistics = load_arrays(root / "Data/training/routing.npz")
    diagonal = load_arrays(output / "experts/diagonal.npz")
    off = load_arrays(output / "experts/off_diagonal.npz")
    diagonal_recipe = read_json(output / "routing/diagonal.json")
    off_recipe = read_json(output / "routing/off_diagonal.json")
    positions = observations["positions"]
    endpoints, priors, weights = [], [], []
    for scope in range(24):
        fit = np.flatnonzero(np.arange(23) != scope)
        raw = diagonal["raw"][diagonal["scope_index"][scope]][:, positions[fit]]
        truth = observations["truth_diag"][fit]
        prior = np.eye(3)[np.square(raw - truth[None]).sum(1).argmin(0)]
        ridge = diagonal_recipe["choices"][scope][1]
        x = (raw / statistics["diagonal_scale"][None, None]).transpose(2, 1, 0)
        clock = np.linspace(1., 0., 86)[positions[fit]][None, :, None]
        x = np.concatenate([clock * x, (1 - clock) * x], axis=-1)
        target = (truth / statistics["diagonal_scale"][None]).T
        q = np.einsum("ntc,ntd->ncd", x, x) / len(fit) + (ridge / 2) * np.eye(6)
        b = np.einsum("ntc,nt->nc", x, target) / len(fit) + (ridge / 2) * np.tile(prior, (1, 2))
        endpoints.append(endpoint_simplex(q, b).reshape(len(prior), 2, 3))
        priors.append(prior)
        raw = off["raw"][off["scope_index"][scope]][:, positions[fit]]
        choice = off_recipe["scopes"][scope]
        a = choice["expert_choice"]["chronos_alpha"]
        b = choice["expert_choice"]["timesfm_alpha"]
        prior = np.array([(1 - a) * (1 - b), a * (1 - b), b, 0.])
        ridge = choice["edge_choice"][0]
        x = (raw / statistics["off_diagonal_scale"][None, None]).transpose(2, 1, 0)
        target = (observations["truth_off"][fit] / statistics["off_diagonal_scale"][None]).T
        q = np.einsum("ntc,ntd->ncd", x, x) / len(fit) + ridge * np.eye(4)
        b = np.einsum("ntc,nt->nc", x, target) / len(fit) + ridge * prior
        initial = simplex_quadratic(q, b)
        prediction = np.einsum("nk,ktn->tn", initial, raw)
        rmse = np.sqrt((np.square(prediction - observations["truth_off"][fit]).sum(-1)
                        + observations["off_background"][fit]) / 2177100) / .0176
        date_weights = 1 / np.maximum(rmse, 1e-12)
        date_weights = np.clip(date_weights / date_weights.mean(), .25, 4)
        q = np.einsum("t,ntc,ntd->ncd", date_weights, x, x) / len(fit) + ridge * np.eye(4)
        b = np.einsum("t,ntc,nt->nc", date_weights, x, target) / len(fit) + ridge * prior
        weights.append(simplex_quadratic(q, b))
    np.savez_compressed(output / "routing/diagonal.npz", weights=np.stack(endpoints), priors=np.stack(priors))
    np.savez_compressed(output / "routing/off_diagonal.npz", weights=np.stack(weights),
                        groups=statistics["off_diagonal_groups"], scale=statistics["off_diagonal_scale"])
    print("Fitted diagonal and off-diagonal routing for all 24 scopes.", flush=True)
