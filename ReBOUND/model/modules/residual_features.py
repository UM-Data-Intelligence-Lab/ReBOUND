from datetime import datetime

import numpy as np


def expert_features(prediction, experts, priors, dates, role):
    days, edges = prediction.shape
    log_prediction = np.log1p(prediction)
    weekday = np.array([datetime.strptime(str(int(day)), "%Y%m%d").weekday() for day in dates])
    columns = [log_prediction, *[np.log1p(expert) - log_prediction for expert in experts],
               *[np.broadcast_to(priors[:, k], (days, edges)) for k in range(3)],
               np.broadcast_to(np.sin(2 * np.pi * weekday / 7)[:, None], (days, edges)),
               np.broadcast_to(np.cos(2 * np.pi * weekday / 7)[:, None], (days, edges)),
               np.broadcast_to(role[:, None], (days, edges))]
    return np.stack(columns, axis=-1).astype(np.float32)


def boundary_features(prediction, truth, positions, held):
    output = []
    for day in range(86):
        fit = np.flatnonzero((~np.isin(np.arange(23), held)) & (positions != day))
        residual = truth[fit] - prediction[positions[fit]]
        mean = residual.mean(0)
        distance = np.abs(positions[fit] - day).astype(float)
        weights = np.exp(-.5 * (distance / 3.) ** 2)
        mass = weights.sum()
        local = np.sum(weights[:, None] * (residual - mean), axis=0) / (.1 + mass)
        output.append(np.stack([np.full(len(mean), np.log1p(mass)),
                                np.full(len(mean), np.log1p(distance.min())), mean, local], -1))
    return np.stack(output).astype(np.float32)


def global_features(prediction, expert_log_totals):
    total = np.log1p(prediction.sum(axis=1))
    row = np.column_stack([total, *[expert - total for expert in expert_log_totals]])
    return np.broadcast_to(row[:, None, :], (86, 509, 4)).astype(np.float32)


def pooled_features(prediction, experts, operators):
    parts = []
    for operator in operators:
        prediction_log = np.log1p(prediction @ operator.T)
        expert_log = np.log1p(experts @ operator.T)
        parts.append(np.stack([prediction_log, *[expert_log[k] - prediction_log for k in range(3)]], -1))
    return np.concatenate(parts, -1).astype(np.float32)


def temporal_shape(raw):
    values = np.asarray(raw, dtype=np.float64)
    first = np.empty_like(values)
    second = np.empty_like(values)
    first[1:-1] = (values[2:] - values[:-2]) / 2
    first[0] = (-3 * values[0] + 4 * values[1] - values[2]) / 2
    first[-1] = (3 * values[-1] - 4 * values[-2] + values[-3]) / 2
    second[1:-1] = values[2:] - 2 * values[1:-1] + values[:-2]
    second[0] = values[2] - 2 * values[1] + values[0]
    second[-1] = values[-1] - 2 * values[-2] + values[-3]
    return np.concatenate([first, second], axis=-1).astype(np.float32)


def features34(features26):
    difference = temporal_shape(features26[:, :, :4])
    mean = difference.mean(axis=0, dtype=np.float64).astype(np.float32)
    return np.concatenate([features26, np.broadcast_to(mean, (86, 509, 8))], axis=-1)
