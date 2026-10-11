from __future__ import annotations



class PeriodicityBasedExperts:
    def __init__(self, diagonal_bank, off_diagonal_bank):
        self.diagonal_bank = diagonal_bank
        self.off_diagonal_bank = off_diagonal_bank

    def predict(self, scope):
        diagonal = self.diagonal_bank["scope_index"][scope]
        off_diagonal = self.off_diagonal_bank["scope_index"][scope]
        return self.diagonal_bank["raw"][diagonal, 0], self.off_diagonal_bank["raw"][off_diagonal, 0]


def build_periodicity_network(component, arrays, config):
    import torch
    from torch import nn
    from typing import Any, Mapping

    class DiagResidualMLP(nn.Module):
        def __init__(self, arrays: FoldArrays, config: Mapping[str, Any]) -> None:
            super().__init__()
            node_dim = int(config["node_embedding_dim"])
            municipality_dim = int(config["municipality_embedding_dim"])
            weekday_dim = int(config["weekday_embedding_dim"])
            day_hidden = int(config["day_hidden_dim"])
            hidden = [int(value) for value in config["hidden_dims"]]
            dropout = float(config["dropout"])
            self.residual_scale = float(config["diag_residual_scale"])
            self.diag_weekday_models = bool(config.get("diag_weekday_models", False))

            self.node_embedding = nn.Embedding(1476, node_dim)
            municipality_count = int(arrays.municipality_by_node.max()) + 1
            self.municipality_embedding = nn.Embedding(
                municipality_count, municipality_dim
            )
            self.weekday_embedding = nn.Embedding(7, weekday_dim)
            self.day_encoder = nn.Sequential(
                nn.Linear(arrays.day_features.shape[1], day_hidden),
                nn.SiLU(),
                nn.Linear(day_hidden, day_hidden),
                nn.SiLU(),
            )
            input_dim = (
                node_dim
                + municipality_dim
                + weekday_dim
                + day_hidden
                + arrays.node_static.shape[1]
                + 1
            )
            def make_network() -> nn.Sequential:
                layers: list[nn.Module] = []
                previous = input_dim
                for width in hidden:
                    layers.extend(
                        [nn.Linear(previous, width), nn.SiLU(), nn.Dropout(dropout)]
                    )
                    previous = width
                output = nn.Linear(previous, 1)
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
                layers.append(output)
                return nn.Sequential(*layers)

            if self.diag_weekday_models:
                self.weekday_networks = nn.ModuleList(
                    [make_network() for _ in range(7)]
                )
            else:
                self.network = make_network()

            active_nodes = torch.as_tensor(arrays.active_nodes, dtype=torch.long)
            municipality = torch.as_tensor(
                arrays.municipality_by_node[arrays.active_nodes], dtype=torch.long
            )
            static = torch.as_tensor(
                arrays.node_static[arrays.active_nodes], dtype=torch.float32
            )
            self.register_buffer("active_nodes", active_nodes)
            self.register_buffer("municipality", municipality)
            self.register_buffer("static", static)

        def forward(
            self,
            day_features: torch.Tensor,
            weekdays: torch.Tensor,
            baseline: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, entity_count = baseline.shape
            day_encoded = self.day_encoder(day_features)
            day_encoded = day_encoded[:, None, :].expand(-1, entity_count, -1)
            weekday = self.weekday_embedding(weekdays)
            weekday = weekday[:, None, :].expand(-1, entity_count, -1)
            node = self.node_embedding(self.active_nodes)
            node = node[None, :, :].expand(batch_size, -1, -1)
            municipality = self.municipality_embedding(self.municipality)
            municipality = municipality[None, :, :].expand(batch_size, -1, -1)
            static = self.static[None, :, :].expand(batch_size, -1, -1)
            baseline_log = torch.log1p(baseline).unsqueeze(-1)
            features = torch.cat(
                (day_encoded, weekday, node, municipality, static, baseline_log),
                dim=-1,
            )
            if self.diag_weekday_models:
                raw_delta = features.new_zeros((batch_size, entity_count))
                for weekday_id, network in enumerate(self.weekday_networks):
                    selected = torch.nonzero(
                        weekdays == weekday_id, as_tuple=False
                    ).squeeze(-1)
                    if selected.numel() == 0:
                        continue
                    selected_delta = network(
                        features.index_select(0, selected)
                    ).squeeze(-1)
                    raw_delta = raw_delta.index_copy(0, selected, selected_delta)
            else:
                raw_delta = self.network(features).squeeze(-1)
            delta = self.residual_scale * torch.tanh(raw_delta)
            return baseline * torch.exp(delta)

    class OffHurdleMLP(nn.Module):
        def __init__(self, arrays: FoldArrays, config: Mapping[str, Any]) -> None:
            super().__init__()
            node_dim = int(config["node_embedding_dim"])
            municipality_dim = int(config["municipality_embedding_dim"])
            edge_dim = int(config["edge_embedding_dim"])
            weekday_dim = int(config["weekday_embedding_dim"])
            day_hidden = int(config["day_hidden_dim"])
            hidden = [int(value) for value in config["hidden_dims"]]
            dropout = float(config["dropout"])
            self.probability_residual_scale = float(config["off_probability_scale"])
            self.magnitude_residual_scale = float(config["off_magnitude_scale"])

            self.node_embedding = nn.Embedding(1476, node_dim)
            municipality_count = int(arrays.municipality_by_node.max()) + 1
            self.municipality_embedding = nn.Embedding(
                municipality_count, municipality_dim
            )
            self.edge_embedding = nn.Embedding(len(arrays.candidate_edges), edge_dim)
            self.weekday_embedding = nn.Embedding(7, weekday_dim)
            self.day_encoder = nn.Sequential(
                nn.Linear(arrays.day_features.shape[1], day_hidden),
                nn.SiLU(),
                nn.Linear(day_hidden, day_hidden),
                nn.SiLU(),
            )
            input_dim = (
                node_dim * 2
                + municipality_dim * 2
                + edge_dim
                + weekday_dim
                + day_hidden
                + arrays.edge_static.shape[1]
                + 2
            )
            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden:
                layers.extend(
                    [nn.Linear(previous, width), nn.SiLU(), nn.Dropout(dropout)]
                )
                previous = width
            self.trunk = nn.Sequential(*layers)
            self.probability_head = nn.Linear(previous, 1)
            self.magnitude_head = nn.Linear(previous, 1)
            nn.init.zeros_(self.probability_head.weight)
            nn.init.zeros_(self.probability_head.bias)
            nn.init.zeros_(self.magnitude_head.weight)
            nn.init.zeros_(self.magnitude_head.bias)

            origin = torch.as_tensor(arrays.edge_origin, dtype=torch.long)
            destination = torch.as_tensor(arrays.edge_destination, dtype=torch.long)
            origin_municipality = torch.as_tensor(
                arrays.municipality_by_node[arrays.edge_origin], dtype=torch.long
            )
            destination_municipality = torch.as_tensor(
                arrays.municipality_by_node[arrays.edge_destination], dtype=torch.long
            )
            edge_static = torch.as_tensor(arrays.edge_static, dtype=torch.float32)
            edge_ids = torch.arange(len(arrays.candidate_edges), dtype=torch.long)
            self.register_buffer("origin", origin)
            self.register_buffer("destination", destination)
            self.register_buffer("origin_municipality", origin_municipality)
            self.register_buffer("destination_municipality", destination_municipality)
            self.register_buffer("edge_static", edge_static)
            self.register_buffer("edge_ids", edge_ids)

        def forward(
            self,
            day_features: torch.Tensor,
            weekdays: torch.Tensor,
            base_probability: torch.Tensor,
            base_positive_mean: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            batch_size, entity_count = base_probability.shape
            day_encoded = self.day_encoder(day_features)
            day_encoded = day_encoded[:, None, :].expand(-1, entity_count, -1)
            weekday = self.weekday_embedding(weekdays)
            weekday = weekday[:, None, :].expand(-1, entity_count, -1)
            origin = self.node_embedding(self.origin)
            destination = self.node_embedding(self.destination)
            origin_municipality = self.municipality_embedding(
                self.origin_municipality
            )
            destination_municipality = self.municipality_embedding(
                self.destination_municipality
            )
            edge = self.edge_embedding(self.edge_ids)
            static_features = torch.cat(
                (
                    origin,
                    destination,
                    origin_municipality,
                    destination_municipality,
                    edge,
                    self.edge_static,
                ),
                dim=-1,
            )
            static_features = static_features[None, :, :].expand(batch_size, -1, -1)
            probability = base_probability.clamp(1e-4, 1.0 - 1e-4)
            probability_logit = torch.logit(probability)
            magnitude_log = torch.log1p(base_positive_mean)
            features = torch.cat(
                (
                    day_encoded,
                    weekday,
                    static_features,
                    probability_logit.unsqueeze(-1),
                    magnitude_log.unsqueeze(-1),
                ),
                dim=-1,
            )
            hidden = self.trunk(features)
            probability_delta = self.probability_residual_scale * torch.tanh(
                self.probability_head(hidden).squeeze(-1)
            )
            magnitude_delta = self.magnitude_residual_scale * torch.tanh(
                self.magnitude_head(hidden).squeeze(-1)
            )
            predicted_probability = torch.sigmoid(
                probability_logit + probability_delta
            )
            predicted_magnitude = base_positive_mean * torch.exp(magnitude_delta)
            prediction = predicted_probability * predicted_magnitude
            return prediction, predicted_probability, predicted_magnitude

    if component not in ("diag", "off"):
        raise ValueError("component must be diag or off")
    return (DiagResidualMLP if component == "diag" else OffHurdleMLP)(arrays, config)
