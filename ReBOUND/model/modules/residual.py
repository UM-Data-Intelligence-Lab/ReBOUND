from pathlib import Path

import joblib
import numpy as np

from .residual_features import (
    boundary_features, expert_features, features34, global_features, pooled_features,
)


def mode_coefficients(truth, prediction, basis, positions, fit, kind, length, noise):
    projected = (truth - prediction[positions]) @ basis
    mean = projected[fit].mean(0)
    if kind == "constant":
        return np.tile(mean, (len(prediction), 1))
    distance = np.arange(len(prediction))[:, None] - positions[fit][None]
    kernel = np.exp(-.5 * np.square(distance / length))
    weights = np.linalg.solve(kernel[positions[fit]] + noise * np.eye(len(fit)), projected[fit] - mean)
    return mean + kernel @ weights


class ProgressiveResidualLearning:
    def __init__(self, directory, state, recipe, observations, expert_predictions):
        self.directory = Path(directory)
        self.state = state
        self.recipe = recipe
        self.positions = state["positions"]
        self.edge_positions = state["parent_positions"]
        self.truth = np.ascontiguousarray(observations["truth_off"][:, self.edge_positions])
        self.experts = expert_predictions[:, :, self.edge_positions]
        self.expert_totals = np.log1p(self.experts.sum(2))
        self.role = np.isin(state["dates"], observations["dates"]).astype(float)
        shared = joblib.load(self.directory / "shared.joblib")
        self.models = {stage: shared[stage] for stage in [
            "low_flow", "expert", "boundary", "global", "pooled", "multiwindow",
        ]}
        self.teacher = shared["teacher"]

    def _features(self, prediction, held, columns):
        result = expert_features(prediction, self.experts, self.state["priors"], self.state["dates"], self.role)
        if columns >= 14:
            result = np.concatenate([result, boundary_features(prediction, self.truth, self.positions, held)], -1)
        if columns >= 18:
            result = np.concatenate([result, global_features(prediction, self.expert_totals)], -1)
        if columns == 26:
            pooled = pooled_features(prediction[None], self.experts, self.state["operators"])[0, self.state["permutation"]]
            result = np.concatenate([result, pooled], -1)
        return result

    def _mode_stack(self, base, fit):
        prediction = base.copy()
        edges = self.edge_positions
        prediction[:, edges] = np.maximum(prediction[:, edges] + self.state["delta"], 0)
        for layer in self.recipe["layers"]:
            config = layer["config"]
            basis = self.state[layer["basis"]]
            if layer["stage"] == "extra_modes":
                basis = basis[:, 4:8]
                residual = self.truth - prediction[self.positions][:, edges]
                coefficient = np.tile((residual[fit] @ basis).mean(0), (86, 1))
            elif layer["stage"] == "constant_modes":
                residual = self.truth - prediction[self.positions][:, edges]
                projected = residual @ basis
                coefficient = np.tile(projected[fit].mean(0), (86, 1))
            else:
                coefficient = mode_coefficients(self.truth, prediction[:, edges], basis, self.positions,
                                                fit, "temporal", config["length"], config["noise"])
            prediction[:, edges] = np.maximum(prediction[:, edges] + config["shrink"] * (coefficient @ basis.T), 0)
        return prediction

    def _crossfit_correction(self, base, fit):
        # Each conditional residual is formed after excluding that same date.
        residual = np.zeros_like(self.truth)
        for day in fit:
            prediction = self._mode_stack(base, fit[fit != day])
            residual[day] = self.truth[day] - prediction[self.positions[day], self.edge_positions]
        config = self.recipe["crossfit"]
        basis = self.state["U4"]
        projected = residual @ basis
        mean = projected[fit].mean(0)
        if config["kind"] == "constant":
            coefficient = np.tile(mean, (86, 1))
        else:
            distance = np.arange(86)[:, None] - self.positions[fit][None]
            kernel = np.exp(-.5 * np.square(distance / config["length"]))
            weights = np.linalg.solve(kernel[self.positions[fit]] + config["noise"] * np.eye(len(fit)), projected[fit] - mean)
            coefficient = mean + kernel @ weights
        return config["shrink"] * (coefficient @ basis.T)

    @staticmethod
    def _predict(model, features):
        return model.predict(features.reshape(-1, features.shape[-1]).astype(float)).reshape(86, 509)

    def _ensemble(self, features, models):
        predictions = []
        for model in models:
            predictions.append(self._predict(model, features))
        return np.stack(predictions).mean(axis=0)

    def prefix(self, base, scope, trace=None):
        trace = trace or (lambda *args: None)
        held = () if scope == 23 else (scope,)
        fit = np.flatnonzero(np.arange(23) != scope)
        edges = self.edge_positions
        whole = self._mode_stack(base, fit)
        whole[:, edges] = np.maximum(whole[:, edges] + self._crossfit_correction(base, fit), 0)
        prediction = whole[:, edges]
        trace("conditional_modes", prediction)
        for stage, strength, columns in [("low_flow", 1., 10), ("expert", .25, 10), ("boundary", .1, 14)]:
            features = self._features(prediction, held, columns)
            delta = self._predict(self.models[stage], features)
            if stage == "low_flow":
                delta = (self.state["priors"][:, 0] <= .8)[None] * delta
            prediction = np.maximum(prediction + strength * delta, 0)
            trace(stage + "_tree", prediction)
        return whole, prediction

    def predict(self, base, scope, trace=None):
        trace = trace or (lambda *args: None)
        scope_models = joblib.load(self.directory / f"scope_{scope:02d}.joblib")
        held = () if scope == 23 else (scope,)
        edges = self.edge_positions
        whole, prediction = self.prefix(base, scope, trace)
        boundary_model = scope_models["boundary"]
        prediction = np.maximum(prediction + .07 * self._predict(boundary_model, self._features(prediction, held, 14)), 0)
        trace("conditional_boundary", prediction)
        for stage, strength, columns in [("global", .25, 18), ("pooled", .7, 26), ("multiwindow", .35, 26)]:
            prediction = np.maximum(prediction + strength * self._predict(self.models[stage], self._features(prediction, held, columns)), 0)
            trace(stage + "_tree", prediction)
        features = features34(self._features(prediction, held, 26))
        prediction = np.maximum(prediction + .45 * self._ensemble(features, scope_models["residual_ensemble"]), 0)
        trace("residual_ensemble", prediction)
        features = features34(self._features(prediction, held, 26))
        delta = self._ensemble(features, scope_models["temporal_ensemble"])
        constant = delta.mean(axis=0, dtype=np.float64)
        strength = .95 * (np.arange(86, dtype=float) / 85)
        prediction = np.maximum(prediction + strength[:, None] * constant[None, :], 0)
        trace("temporal_ensemble", prediction)
        features = features34(self._features(prediction, held, 26))
        prediction = np.maximum(prediction + .1 * self._predict(self.teacher, features), 0)
        whole[:, edges] = prediction
        trace("teacher", prediction)
        return whole
