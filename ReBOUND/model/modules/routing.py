import numpy as np


class BoundaryAnchoredExpertRouting:
    def __init__(self, diagonal, off_diagonal, diagonal_recipe, off_diagonal_recipe):
        self.diagonal = diagonal
        self.off_diagonal = off_diagonal
        self.diagonal_recipe = diagonal_recipe
        self.off_diagonal_recipe = off_diagonal_recipe

    def predict(self, diagonal_experts, off_diagonal_experts, scope):
        # Keep the operation order of the released expert mixture.
        choice = self.off_diagonal_recipe["scopes"][scope]
        selected = choice["expert_choice"]
        mlp, chronos, timesfm = off_diagonal_experts[:3]
        alpha, beta = selected["chronos_alpha"], selected["timesfm_alpha"]
        mixed = mlp if alpha == 0 else (1 - alpha) * mlp + alpha * chronos
        global_prediction = mixed if beta == 0 else (1 - beta) * mixed + beta * timesfm
        fractions = choice["edge_choice"][1:]
        if any(fractions):
            fraction = np.asarray(fractions)[self.off_diagonal["groups"]][None]
            local_prediction = np.einsum("nk,ktn->tn", self.off_diagonal["weights"][scope], off_diagonal_experts)
            off = (1 - fraction) * global_prediction + fraction * local_prediction
        else:
            off = global_prediction.copy()

        kind, _, fraction = self.diagonal_recipe["choices"][scope]
        if kind == "identity":
            diag = diagonal_experts[0].copy()
        else:
            hard = np.einsum("nk,ktn->tn", self.diagonal["priors"][scope], diagonal_experts)
            if kind == "hard":
                diag = hard
            else:
                endpoints = self.diagonal["weights"][scope]
                clock = np.linspace(1., 0., diagonal_experts.shape[1])
                weights = clock[:, None, None] * endpoints[None, :, 0, :] + (1 - clock[:, None, None]) * endpoints[None, :, 1, :]
                convex = np.einsum("tnk,ktn->tn", weights, diagonal_experts)
                diag = (1 - fraction) * hard + fraction * convex
        return diag, off
