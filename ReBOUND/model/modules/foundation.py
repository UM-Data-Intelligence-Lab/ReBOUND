import numpy as np


def recovery_weight(evacuee_fraction, water_recovery, anchor=0):
    severity = np.maximum(np.nan_to_num(evacuee_fraction) * (1 - np.nan_to_num(water_recovery)), 0.)
    return np.zeros_like(severity) if severity[anchor] <= 1e-12 else np.clip(severity / severity[anchor], 0, 1)


def bidirectional_blend(forward, backward, weight):
    return weight[:, None] * forward + (1 - weight[:, None]) * backward


def forecast_chronos(pipeline, context, future, target_names):
    output = pipeline.predict_df(
        context, future_df=future, id_column="item_id", timestamp_column="timestamp",
        target=target_names, prediction_length=len(future), quantile_levels=[0.5],
        batch_size=256, cross_learning=True, validate_inputs=True, freq="D",
    )
    pivot = output.pivot(index="timestamp", columns="target_name", values="predictions")
    result = pivot.reindex(columns=target_names).to_numpy(dtype=np.float64)
    if result.shape != (len(future), len(target_names)) or not np.isfinite(result).all():
        raise ValueError("Unexpected Chronos prediction shape or values.")
    return np.maximum(result, 0.)


class MobilityAdaptedFoundationExperts:
    def __init__(self, diagonal_bank, off_diagonal_bank):
        self.diagonal_bank = diagonal_bank
        self.off_diagonal_bank = off_diagonal_bank

    def predict(self, scope):
        diagonal = self.diagonal_bank["scope_index"][scope]
        off_diagonal = self.off_diagonal_bank["scope_index"][scope]
        return self.diagonal_bank["raw"][diagonal, 1:], self.off_diagonal_bank["raw"][off_diagonal, 1:]
