from datetime import date, datetime, timedelta
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ..model.modules.foundation import forecast_chronos
from ..utils.io import load_arrays, read_json, write_json
from .experts import setup, profile


def frame(days, targets, covariates, target_names, covariate_names):
    data = {"item_id": ["panel"] * len(days), "timestamp": pd.to_datetime(days)}
    data.update({name: targets[:, i] for i, name in enumerate(target_names)})
    data.update({name: covariates[:, i] for i, name in enumerate(covariate_names)})
    return pd.DataFrame(data)


def inference_frames(data, positions):
    days = [datetime.strptime(str(int(d)), "%Y%m%d").date() for d in data["calendar_dates"]]
    gap = data["gap"]
    gap_days = [d for d, keep in zip(days, gap) if keep]
    left = np.asarray([d < gap_days[0] for d in days])
    right = np.asarray([d > gap_days[-1] for d in days])
    names = [f"edge_{i:03d}" for i in range(len(positions))]
    cov_names = data["covariate_names"].astype(str).tolist()
    target, cov = data["visible_off"][:, positions], data["covariates"]
    forward = frame([d for d, keep in zip(days, left) if keep], target[left], cov[left], names, cov_names)
    future = frame(gap_days, None, cov[gap], [], cov_names)
    synthetic = date(2000, 1, 1)
    backward_days = [synthetic + timedelta(days=i) for i in range(int(right.sum()))]
    backward_future = [synthetic + timedelta(days=int(right.sum()) + i) for i in range(int(gap.sum()))]
    backward = frame(backward_days, target[right][::-1], cov[right][::-1], names, cov_names)
    future_backward = frame(backward_future, None, cov[gap][::-1], [], cov_names)
    return (forward, future, backward, future_backward), names, (left, right)


def train_foundation_group(root, output, group, model_path=None, device="cuda:0", gpu_memory_gib=5):
    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts
    from transformers import TrainerCallback
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_arrays(root / "Data/training/experts/target/foundation.npz", allow_missing=("visible_off", "visible_diag"))
    if data["observed"][data["gap"]].any() or not np.isnan(data["visible_off"][~data["observed"]]).all():
        raise ValueError("Foundation training requires masked gap targets.")
    config = read_json(root / "ReBOUND/config/foundation.json")
    device = setup(config["seed"], device, gpu_memory_gib)
    started = time.monotonic()
    positions = np.flatnonzero(data["group_id"] == group)
    frames, names, sides = inference_frames(data, positions)
    cov_names = data["covariate_names"].astype(str).tolist()
    past_only = [name for name in cov_names if name.startswith("full_diag_pc_")]
    known = [name for name in cov_names if name not in past_only]
    arguments = dict(device_map=str(device), torch_dtype=torch.bfloat16)
    if model_path is None:
        model_path = config["model"]
        arguments["revision"] = config["revision"]
    elif Path(model_path).exists():
        arguments["local_files_only"] = True
    pipeline = Chronos2Pipeline.from_pretrained(model_path, **arguments)
    result = pipeline.predict_df(frames[0], future_df=frames[1], id_column="item_id", timestamp_column="timestamp",
        target=names, prediction_length=len(frames[1]), quantile_levels=config["quantiles"],
        batch_size=256, cross_learning=True, validate_inputs=True, freq="D")
    median = np.maximum(result.pivot(index="timestamp", columns="target_name", values="predictions").reindex(columns=names).to_numpy(dtype=float), 0)
    quantiles = np.ascontiguousarray(result[[str(q) for q in config["quantiles"]]].to_numpy(dtype=float))
    result["predictions"] = (np.sort(np.maximum(quantiles, 0), axis=-1) * np.asarray(config["quadrature_weights"])).sum(-1)
    mean = result.pivot(index="timestamp", columns="target_name", values="predictions").reindex(columns=names).to_numpy(dtype=float)
    forward, future, backward, future_backward = frames
    future = future.drop(columns=past_only)
    future_backward = future_backward.drop(columns=past_only)
    trained = []

    class SaveAdapter(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if args.max_steps != 200 or args.dataloader_num_workers != 0:
                raise ValueError("Unexpected foundation training schedule.")
            self.initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if int(state.global_step) != config["retained_step"]:
                return control
            model.eval()
            adapted = Chronos2Pipeline(model=model)
            with torch.no_grad():
                f = forecast_chronos(adapted, forward, future, names)
                b = forecast_chronos(adapted, backward, future_backward, names)[::-1]
            trained.append(np.maximum(data["route"][:, None] * f + (1 - data["route"][:, None]) * b, 0).astype(np.float32))
            changed = any(not torch.equal(self.initial[name], value.detach().cpu()) for name, value in model.named_parameters() if value.requires_grad)
            if not changed:
                raise RuntimeError("LoRA parameters did not change.")
            model.save_pretrained(output / "adapter")
            write_json(output / "profile.json", profile(device, started, train_and_inference_measured=True, weights_changed=changed))
            control.should_training_stop = True
            return control

    eligible = np.flatnonzero(data["fit_rank"][positions] > 0)
    values = data["visible_off"][:, positions]
    inputs = [dict(target=values[side][:, eligible].T.astype(np.float32),
        past_covariates={name: data["covariates"][side, i].astype(np.float32) for i, name in enumerate(cov_names)}) for side in sides]
    prepared = from_list_of_dicts(inputs, prediction_length=28, known_covariates_names=known,
                                  use_target_encoding=False, validate_inputs=True)
    pipeline.fit(prepared, prediction_length=28, finetune_mode="lora", lora_config=config["lora"],
        context_length=64, learning_rate=1e-5, num_steps=200, batch_size=256, output_dir=output / "trainer",
        min_past=28, callbacks=[SaveAdapter()], disable_tqdm=True, remove_printer_callback=True,
        seed=config["seed"], data_seed=config["seed"], logging_steps=10, report_to="none")
    if len(trained) != 1:
        raise RuntimeError("Retained LoRA checkpoint was not produced.")
    np.savez_compressed(output / "predictions.npz", positions=positions, adapted=trained[0],
                        median=median.astype(np.float32), quantile_mean=mean.astype(np.float32))
    write_json(output / "training.json", profile(device, started, group=group, lora_steps=config["retained_step"],
        schedule_steps=200, train_dates=data["calendar_dates"][data["observed"]].tolist(),
        foundation_model=config["model"], revision=config["revision"], past_only_covariates=past_only))


def assemble_foundation(directory, weights):
    directory, weights = Path(directory), Path(weights)
    off = load_arrays(weights / "experts/off_diagonal.npz")
    features = load_arrays(weights / "experts/residual_features.npz")
    seen = np.zeros(off["raw"].shape[-1], dtype=int)
    clock = (np.arange(86, dtype=float) / 85)[:, None]
    for group in range(4):
        data = load_arrays(directory / f"group_{group}" / "predictions.npz")
        positions = data["positions"]
        for raw in off["raw"]:
            raw[1][:, positions] = data["adapted"]
            raw[3][:, positions] = (1 - clock) * data["adapted"].astype(float) + clock * data["median"].astype(float)
        features["predictions"][1][:, positions] = data["quantile_mean"]
        seen[positions] += 1
    if not np.all(seen == 1):
        raise ValueError("Foundation groups must cover every candidate edge exactly once.")
    np.savez_compressed(weights / "experts/off_diagonal.npz", **off)
    np.savez_compressed(weights / "experts/residual_features.npz", **features)
