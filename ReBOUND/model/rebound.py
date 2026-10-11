from pathlib import Path

import numpy as np

from ..utils.io import load_arrays, read_json
from .modules.periodicity import PeriodicityBasedExperts
from .modules.foundation import MobilityAdaptedFoundationExperts
from .modules.grouped import FlowMagnitudeGroupedExperts
from .modules.routing import BoundaryAnchoredExpertRouting
from .modules.boundary_calibration import BoundaryCalibration
from .modules.residual import ProgressiveResidualLearning


class ReBOUND:
    def __init__(self, root, pretrained=None, residual=True):
        root = Path(root)
        pretrained = Path(pretrained) if pretrained is not None else root / "Pretrained"
        self.calendar = load_arrays(root / "Data/calendar.npz")
        self.observations = load_arrays(root / "Data/observations.npz")
        diagonal_bank = load_arrays(pretrained / "experts/diagonal.npz")
        off_diagonal_bank = load_arrays(pretrained / "experts/off_diagonal.npz")
        self.periodicity = PeriodicityBasedExperts(diagonal_bank, off_diagonal_bank)
        self.foundation = MobilityAdaptedFoundationExperts(diagonal_bank, off_diagonal_bank)
        diagonal = load_arrays(pretrained / "routing/diagonal.npz")
        off_diagonal = load_arrays(pretrained / "routing/off_diagonal.npz")
        diagonal_recipe = read_json(pretrained / "routing/diagonal.json")
        off_diagonal_recipe = read_json(pretrained / "routing/off_diagonal.json")
        self.router = BoundaryAnchoredExpertRouting(diagonal, off_diagonal, diagonal_recipe, off_diagonal_recipe)
        self.boundary = BoundaryCalibration(self.observations, off_diagonal["scale"], off_diagonal_recipe, diagonal_recipe)
        self.grouped = FlowMagnitudeGroupedExperts(load_arrays(pretrained / "experts/grouped.npz"), off_diagonal_recipe["grouped"])
        self.spatial = load_arrays(pretrained / "experts/spatial.npz")["diagonal"]
        self.spatial_recipe = read_json(pretrained / "routing/spatial.json")
        self.residual = ProgressiveResidualLearning(
            pretrained / "residual", load_arrays(pretrained / "residual/basis.npz"),
            read_json(pretrained / "residual/recipe.json"), self.observations,
            load_arrays(pretrained / "experts/residual_features.npz")["predictions"],
        ) if residual else None

    def base_prediction(self, scope=23, trace=None):
        if scope not in range(24):
            raise ValueError("Scope must be an integer between 0 and 23.")
        trace = trace or (lambda *args: None)
        periodic_diag, periodic_off = self.periodicity.predict(scope)
        temporal_diag, temporal_off = self.foundation.predict(scope)
        diag_experts = np.concatenate([periodic_diag[None], temporal_diag])
        off_experts = np.concatenate([periodic_off[None], temporal_off])
        diag, off = self.router.predict(diag_experts, off_experts, scope)
        trace("routed_diag", diag)
        trace("routed_off", off)
        diag = self.boundary.diagonal(diag, scope)
        off = self.boundary.off_diagonal(off, scope)
        trace("calibrated_diag", diag)
        trace("calibrated_off", off)
        off = self.grouped.apply(off, scope)
        trace("grouped_off", off)
        alpha = self.spatial_recipe[scope]["alpha"]
        diag = diag + alpha * (self.spatial[scope] - diag)
        trace("spatial_diag", diag)
        return diag, off

    def predict(self, scope=23, trace=None):
        diag, off = self.base_prediction(scope, trace)
        off = self.residual.predict(off, scope, trace)
        for component in [diag, off]:
            if not np.isfinite(component).all() or np.any(component < 0):
                raise ValueError("Invalid ReBOUND prediction.")
        return dict(prediction_dates=self.calendar["prediction_dates"], active_nodes=self.calendar["active_nodes"],
                    edges=self.calendar["edges"], candidate_diag=diag, candidate_off=off)
