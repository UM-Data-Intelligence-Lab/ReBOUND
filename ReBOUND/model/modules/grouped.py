import numpy as np


def adjustment_ratio(coefficients, occurrence_prior):
    return np.clip((coefficients + .02) / (occurrence_prior[None] + .02), .5, 2)


class FlowMagnitudeGroupedExperts:
    def __init__(self, arrays, selections):
        self.arrays = arrays
        self.selections = selections

    def apply(self, prediction, scope):
        prediction = prediction.copy()
        clock = np.arange(len(prediction), dtype=float) / (len(prediction) - 1)
        # Retained order: ranks 51-150, 151-250, then 1-50.
        for group, selections in enumerate(self.selections):
            choice = selections[scope]
            left = choice.get("alpha_left", choice.get("alpha"))
            right = choice.get("alpha_right", choice.get("alpha"))
            if left > 0 or right > 0:
                step = int(np.flatnonzero(self.arrays[f"checkpoints_{group}"] == choice["checkpoint"])[0])
                weight = left if left == right else ((1 - clock) * left + clock * right)[:, None]
                positions = self.arrays[f"positions_{group}"]
                ratio = self.arrays[f"ratios_{group}"][step]
                prediction[:, positions] *= 1 + weight * (ratio - 1)
        return prediction


def build_grouped_network(public_dimensions, config, matched=False):
    import torch
    from torch import nn

    class GroupedFlowMLP(nn.Module):

        def __init__(self, dims, tr, matched):
            super().__init__()
            self.matched = matched
            self.edge = nn.Embedding(tr.get('edge_count', 100), tr['edge_embedding'])
            d = dims + 6 + 1 + tr.get('context_channels', 10) + 10 + 2 + tr['edge_embedding']
            (a, b) = tr['hidden']
            self.net = nn.Sequential(nn.Linear(d, a), nn.SiLU(), nn.Linear(a, b), nn.SiLU(), nn.Linear(b, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, z, v, day, e, temporal):
            prior = z['view_prior'][v, z['weekdays'][day], e]
            logit = torch.logit(prior)
            rates = z['view_rates'][v, e]
            rates = rates if temporal else rates * 0.0
            window = z['view_windows'][v]
            horizon = torch.stack([day - window[:, 0] + 1, window[:, 1] - day + 1], -1).float() / 86.0
            x = torch.cat([z['public'][day], z['edge_static'][e], logit[:, None], rates, z['view_available'][v], horizon, self.edge(e)], -1)
            return logit + self.net(x)[:, 0]

        def flow(self, z, v, d, e, temporal):
            logits = self(z, v, d, e, temporal)
            p = z['view_prior'][v, z['weekdays'][d], e]
            delta = logits - torch.logit(p)
            if self.matched:
                delta = p * (1 - p) / -torch.expm1(-p) * delta
            return nn.functional.softplus(torch.log(torch.expm1(p)) + delta)

    return GroupedFlowMLP(public_dimensions, config, matched)
