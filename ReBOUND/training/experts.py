from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..model.modules.periodicity import build_periodicity_network
from ..model.modules.grouped import build_grouped_network
from ..utils.io import load_arrays, read_json, write_json
from .optimization import set_reproducible, train_epoch_diag, train_epoch_off, predict_diag, predict_off
from .sampling import sampler


def setup(seed, device, gpu_memory_gib):
    set_reproducible(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type == "cuda":
        device = torch.device("cuda", 0 if device.index is None else device.index)
        capacity = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(gpu_memory_gib * 1024**3 / capacity, .95), device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def profile(device, started, **extra):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(seconds=time.monotonic() - started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                peak_gpu_mib=torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else 0,
                **extra)


def train_periodicity(root, output, task, component, recipe, seed, device="cuda", gpu_memory_gib=2):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_arrays(root / "Data/training/experts" / task / "periodicity.npz")
    arrays = SimpleNamespace(**data)
    if np.intersect1d(arrays.train_indices, arrays.valid_indices).size:
        raise ValueError("Training and prediction dates overlap.")
    if np.any(arrays.diag_targets[arrays.valid_indices]) or np.any(arrays.off_targets[arrays.valid_indices]):
        raise ValueError("Gap labels must be masked before expert training.")
    config = read_json(root / "ReBOUND/config/periodicity.json")["recipes"][recipe]
    training = config["training"]
    effective_seed = seed + (10000 if component == "off" else 0)
    device = setup(effective_seed, device, gpu_memory_gib)
    rng = np.random.default_rng(effective_seed)
    model = build_periodicity_network(component, arrays, config["model"]).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    target = getattr(arrays, component + "_targets")
    scale = max(float(np.sqrt(np.mean(target[arrays.train_indices] ** 2))), 1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    train = train_epoch_diag if component == "diag" else train_epoch_off
    predict = predict_diag if component == "diag" else predict_off
    predictions, history = [], []
    started = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        loss = train(model, optimizer, arrays, training, scale, device, rng)
        value = predict(model, arrays, device, config[component + "_blend"], training["batch_days"])
        if not np.isfinite(value).all() or not np.isfinite(loss):
            raise FloatingPointError("Non-finite periodicity training result.")
        predictions.append(value)
        history.append(dict(epoch=epoch, loss=loss))
        if epoch == 5:
            write_json(output / "profile.json", profile(device, started, train_and_inference_measured=True))
        if epoch == 1 or epoch % 10 == 0:
            print(f"{task}/{recipe}/{component}/{seed}: epoch {epoch}, loss {loss:.6f}", flush=True)
    changed = any(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.named_parameters())
    if not changed:
        raise RuntimeError("Optimizer did not change any model parameter.")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / "model.pt")
    np.savez_compressed(output / "predictions.npz", predictions=np.stack(predictions),
                        epochs=np.arange(1, config["epochs"] + 1), prediction_dates=data["prediction_dates"])
    write_json(output / "training.json", profile(device, started, history=history, from_random_initialization=True,
               weights_changed=changed, train_dates=data["dates"][arrays.train_indices].tolist(),
               seed=seed, effective_seed=effective_seed, recipe=recipe, component=component))


def train_grouped(root, output, task, group, loss, seed, device="cuda", gpu_memory_gib=2):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_arrays(root / "Data/training/experts" / task / f"grouped_{group}.npz")
    config = read_json(root / "ReBOUND/config/grouped.json")["groups"][group]
    training = config["training"][loss]
    device = setup(seed, device, gpu_memory_gib)
    model = build_grouped_network(data["public"].shape[1], training, matched=group == "middle").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    z = {k: torch.as_tensor(v, device=device) for k, v in data.items()}
    rng = np.random.default_rng(seed + 70000)
    choose = sampler(data, config, int(data["prediction_dates"][0]), int(data["prediction_dates"][-1]))
    if np.intersect1d(data["train_days"], data["prediction_days"]).size:
        raise ValueError("Grouped expert training includes gap days.")
    temporal = config["arm"] != "context_free"
    count = len(data["selected_edges"])
    started, predictions, history = time.monotonic(), [], []

    def predict():
        model.eval()
        days = z["prediction_days"].repeat_interleave(count)
        edges = torch.arange(count, device=device).repeat(len(z["prediction_days"]))
        views = torch.full_like(days, len(z["view_windows"]) - 1)
        with torch.no_grad():
            result = model.flow(z, views, days, edges, temporal).reshape(-1, count).cpu().numpy()
        model.train()
        return result

    initial = predict()
    np.testing.assert_allclose(initial, data["view_prior"][-1, data["weekdays"][data["prediction_days"]]], rtol=0, atol=2e-7)
    for step in range(1, training["steps"] + 1):
        chosen = rng.integers(0, len(data["train_days"]), size=training["effective_batch"])
        uniform = rng.random(training["effective_batch"])
        views = choose(chosen, uniform)
        days = data["train_days"][chosen]
        edges = rng.integers(0, count, size=training["effective_batch"])
        v, d, e = torch.as_tensor(np.stack([views, days, edges]), device=device)
        optimizer.zero_grad(set_to_none=True)
        coefficient = model.flow(z, v, d, e, temporal)
        mu = coefficient * z["view_positive_mean"][v, z["weekdays"][d], e]
        y = z["count"][d, e]
        if loss == "tweedie":
            safe = mu + 1e-6
            objective = ((torch.sqrt(y) - torch.sqrt(safe)).square() / torch.sqrt(safe)).mean() / float(data["loss_normalizer"]) ** .25
        else:
            objective = torch.square(mu - y).mean() / float(data["loss_normalizer"])
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite grouped training loss.")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 20:
            predict()
            write_json(output / "profile.json", profile(device, started, train_and_inference_measured=True))
        if step in training["checkpoints"]:
            predictions.append(predict())
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / f"step_{step}.pt")
        if step % 300 == 0:
            value = float(objective.detach())
            history.append(dict(step=step, loss=value))
            print(f"{task}/{group}/{loss}/{seed}: step {step}, loss {value:.6f}", flush=True)
    if np.array_equal(initial, predictions[-1]):
        raise RuntimeError("Grouped expert remained at its initialization.")
    np.savez_compressed(output / "predictions.npz", flow_coefficient=np.stack(predictions),
        positive_mean=data["view_positive_mean"][-1, data["weekdays"][data["prediction_days"]]],
        checkpoints=np.asarray(training["checkpoints"]), prior=initial,
        occurrence_prior=data["view_prior"][-1, data["weekdays"][data["prediction_days"]]],
        prediction_dates=data["prediction_dates"], selected_edges=data["selected_edges"])
    write_json(output / "training.json", profile(device, started, history=history, seed=seed,
        from_random_initialization=True, weights_changed=True,
        train_dates=data["calendar"][data["train_days"]].tolist()))
