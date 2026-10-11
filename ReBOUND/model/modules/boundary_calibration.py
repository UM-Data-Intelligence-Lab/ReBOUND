import numpy as np


class BoundaryCalibration:
    def __init__(self, observations, off_scale, off_recipe, diagonal_recipe):
        self.observations = observations
        self.positions = observations["positions"]
        self.left = observations["dates"] < 20240201
        left_end = self.positions[self.left].max()
        right_start = self.positions[~self.left].min()
        self.clock = np.clip((right_start - np.arange(86)) / (right_start - left_end), 0, 1)
        self.distance = np.where(self.left, left_end - self.positions, self.positions - right_start)
        self.off_scale = off_scale
        self.off_recipe = off_recipe
        self.diagonal_recipe = diagonal_recipe

    @staticmethod
    def _fit_indices(scope):
        return np.flatnonzero(np.arange(23) != scope)

    @staticmethod
    def _date_weights(prediction, truth, background, denominator, normalizer):
        rmse = np.sqrt((np.square(prediction - truth).sum(-1) + background) / denominator) / normalizer
        weight = 1 / np.maximum(rmse, 1e-12)
        return np.clip(weight / weight.mean(), .25, 4.)

    def _off_field(self, base, scope, choice):
        ridge, strength, half_life = choice
        fit = self._fit_indices(scope)
        truth = self.observations["truth_off"][fit]
        prediction = base[self.positions[fit]]
        residual = (truth - prediction) / self.off_scale[None]
        recency = np.exp2(-self.distance[fit] / half_life)
        background = self.observations["off_background"][fit]
        weight = self._date_weights(prediction, truth, background, 2177100, .0176)
        clock = self.clock[self.positions[fit], None]
        config = self.off_recipe["calibration"]
        for _ in range(config["date_weight"]["iterations"]):
            total = recency * weight
            wl, wr = total * self.left[fit], total * (~self.left[fit])
            left = (wl @ residual) / (wl.sum() + ridge)
            right = (wr @ residual) / (wr.sum() + ridge)
            if config["metric_reweight"] == "iterative":
                field = clock * left[None] + (1 - clock) * right[None]
                adjusted = np.maximum(prediction + strength * np.clip(field, -config["search"]["cap"], config["search"]["cap"]) * self.off_scale[None], 0)
                weight = self._date_weights(adjusted, truth, background, 2177100, .0176)
        return left, right

    def off_diagonal(self, base, scope):
        left, right = self.off_recipe["scopes"][scope]["calibration_choice"]
        clock = self.clock[:, None]
        cap = self.off_recipe["calibration"]["search"]["cap"]
        if left == right:
            if left[1] == 0:
                return base.copy()
            l, r = self._off_field(base, scope, left)
            field = clock * l[None] + (1 - clock) * r[None]
            return np.maximum(base + left[1] * np.clip(field, -cap, cap) * self.off_scale[None], 0)
        sl, sr = left[1], right[1]
        l = self._off_field(base, scope, left)[0] if sl else np.zeros(base.shape[1])
        r = self._off_field(base, scope, right)[1] if sr else np.zeros(base.shape[1])
        scale = clock * sl + (1 - clock) * sr
        numerator = clock * sl * l[None] + (1 - clock) * sr * r[None]
        field = np.divide(numerator, scale, out=np.zeros_like(numerator), where=scale > 0)
        delta = scale * np.clip(field, -cap, cap) * self.off_scale[None]
        return np.where(scale > 0, np.maximum(base + delta, 0), base)

    def _diagonal_stats(self, base, scope, half_life):
        fit = self._fit_indices(scope)
        truth = self.observations["truth_diag"][fit]
        prediction = base[self.positions[fit]]
        weight = self._date_weights(prediction, truth, self.observations["diag_background"][fit], 1476, 26.57)
        weight = np.exp2(-self.distance[fit] / half_life) * weight
        slope = 1 + prediction
        curvature = np.square(slope)
        curvature /= curvature.mean(axis=0, keepdims=True)
        target = (truth - prediction) / slope
        total = weight[:, None] * curvature
        wl, wr = total * self.left[fit, None], total * (~self.left[fit, None])
        return ((wl * target).sum(axis=0), wl.sum(axis=0), (wr * target).sum(axis=0), wr.sum(axis=0))

    def diagonal(self, base, scope):
        left, right = self.diagonal_recipe["calibration_choices"][scope]
        clock = self.clock[:, None]
        cap = self.diagonal_recipe["cap"]
        if left == right:
            ridge, strength, half_life = left
            if strength == 0:
                return base.copy()
            l, lw, r, rw = self._diagonal_stats(base, scope, half_life)
            delta = strength * np.clip(clock * l[None] / (lw + ridge) + (1 - clock) * r[None] / (rw + ridge), -cap, cap)
            return np.maximum(np.expm1(np.log1p(base) + delta), 0)
        rl, sl, hl = left
        rr, sr, hr = right
        if sl:
            l, lw, _, _ = self._diagonal_stats(base, scope, hl)
            l = l / (lw + rl)
        else:
            l = np.zeros(base.shape[1])
        if sr:
            _, _, r, rw = self._diagonal_stats(base, scope, hr)
            r = r / (rw + rr)
        else:
            r = np.zeros(base.shape[1])
        scale = clock * sl + (1 - clock) * sr
        numerator = clock * sl * l[None] + (1 - clock) * sr * r[None]
        field = np.divide(numerator, scale, out=np.zeros_like(numerator), where=scale > 0)
        value = np.maximum(np.expm1(np.log1p(base) + scale * np.clip(field, -cap, cap)), 0.)
        return np.where(scale > 0, value, base)
