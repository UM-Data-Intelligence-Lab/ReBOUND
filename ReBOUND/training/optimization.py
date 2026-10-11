import math
import random
import numpy as np
import torch
from torch.nn import functional as F

def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def iterate_batches(indices: np.ndarray, batch_days: int, rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return [shuffled[start:start + batch_days] for start in range(0, len(shuffled), batch_days)]

def tensor_rows(array: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array[indices], device=device)

def scaled_raw_loss(prediction: torch.Tensor, target: torch.Tensor, raw_scale: float, mode: str) -> torch.Tensor:
    squared = (prediction - target) ** 2
    if mode == 'mse':
        return squared.mean() / (raw_scale * raw_scale)
    if mode == 'global_rmse':
        return torch.sqrt(squared.mean() + 1e-12) / raw_scale
    if mode == 'daily_rmse':
        return torch.sqrt(squared.mean(dim=1) + 1e-12).mean() / raw_scale
    raise ValueError(f'Unknown raw_loss_mode: {mode}')

def diag_loss(prediction: torch.Tensor, target: torch.Tensor, raw_scale: float, log_weight: float, raw_loss_mode: str='mse') -> torch.Tensor:
    raw = scaled_raw_loss(prediction, target, raw_scale, raw_loss_mode)
    log = F.smooth_l1_loss(torch.log1p(prediction), torch.log1p(target))
    return raw + log_weight * log

def off_loss(prediction: torch.Tensor, probability: torch.Tensor, magnitude: torch.Tensor, target: torch.Tensor, raw_scale: float, log_weight: float, bce_weight: float, magnitude_weight: float, raw_loss_mode: str='mse') -> torch.Tensor:
    raw = scaled_raw_loss(prediction, target, raw_scale, raw_loss_mode)
    log = F.smooth_l1_loss(torch.log1p(prediction), torch.log1p(target))
    occurrence = (target > 0.0).float()
    bce = F.binary_cross_entropy(probability, occurrence)
    positive = target > 0.0
    if positive.any():
        positive_loss = F.smooth_l1_loss(torch.log1p(magnitude[positive]), torch.log1p(target[positive]))
    else:
        positive_loss = torch.zeros((), device=target.device)
    return raw + log_weight * log + bce_weight * bce + magnitude_weight * positive_loss

def train_epoch_diag(model, optimizer, arrays, config, raw_scale, device, rng) -> float:
    model.train()
    losses = []
    for batch in iterate_batches(arrays.train_indices, int(config['batch_days']), rng):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tensor_rows(arrays.day_features, batch, device), tensor_rows(arrays.weekdays, batch, device).long(), tensor_rows(arrays.diag_baseline, batch, device))
        target = tensor_rows(arrays.diag_targets, batch, device)
        loss = diag_loss(prediction, target, raw_scale, float(config['diag_log_weight']), config.get('raw_loss_mode', 'mse'))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['gradient_clip']))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return math.fsum(losses) / len(losses)

def train_epoch_off(model, optimizer, arrays, config, raw_scale, device, rng) -> float:
    model.train()
    losses = []
    for batch in iterate_batches(arrays.train_indices, int(config['batch_days']), rng):
        optimizer.zero_grad(set_to_none=True)
        (prediction, probability, magnitude) = model(tensor_rows(arrays.day_features, batch, device), tensor_rows(arrays.weekdays, batch, device).long(), tensor_rows(arrays.off_probability, batch, device), tensor_rows(arrays.off_positive_mean, batch, device))
        target = tensor_rows(arrays.off_targets, batch, device)
        loss = off_loss(prediction, probability, magnitude, target, raw_scale, float(config['off_log_weight']), float(config['off_bce_weight']), float(config['off_magnitude_weight']), config.get('raw_loss_mode', 'mse'))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['gradient_clip']))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return math.fsum(losses) / len(losses)

def predict_diag(model, arrays, device, blend, batch_days):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(arrays.valid_indices), batch_days):
            batch = arrays.valid_indices[start:start + batch_days]
            base = tensor_rows(arrays.diag_baseline, batch, device)
            raw = model(tensor_rows(arrays.day_features, batch, device), tensor_rows(arrays.weekdays, batch, device).long(), base)
            result.append((base + blend * (raw - base)).cpu().numpy())
    return np.concatenate(result)

def predict_off(model, arrays, device, blend, batch_days):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(arrays.valid_indices), batch_days):
            batch = arrays.valid_indices[start:start + batch_days]
            (raw, _, _) = model(tensor_rows(arrays.day_features, batch, device), tensor_rows(arrays.weekdays, batch, device).long(), tensor_rows(arrays.off_probability, batch, device), tensor_rows(arrays.off_positive_mean, batch, device))
            base = tensor_rows(arrays.off_baseline, batch, device)
            result.append((base + blend * (raw - base)).cpu().numpy())
    return np.concatenate(result)
