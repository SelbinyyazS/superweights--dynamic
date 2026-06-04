import torch
from torch import nn
from torch.nn import functional as F

from src.models.sage_layer import GrowthMode, GrowthStats, MaskedLinear


NeuronStats = dict[str, float]
NeuronPruneMode = str


class SparseMLP(nn.Module):
    """Three-layer sparse MLP for flattened 28x28 images."""

    def __init__(self, hidden_dim: int | tuple[int, int] = 256, sparsity: float = 0.95) -> None:
        super().__init__()
        hidden1_dim, hidden2_dim = self._resolve_hidden_dims(hidden_dim)
        self.layers = nn.ModuleList(
            [
                MaskedLinear(784, hidden1_dim, sparsity=sparsity),
                MaskedLinear(hidden1_dim, hidden2_dim, sparsity=sparsity),
                MaskedLinear(hidden2_dim, 10, sparsity=sparsity),
            ]
        )
        self.register_buffer("hidden1_neuron_mask", torch.ones(hidden1_dim))
        self.register_buffer("hidden2_neuron_mask", torch.ones(hidden2_dim))

    @staticmethod
    def _resolve_hidden_dims(hidden_dim: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(hidden_dim, int):
            return hidden_dim, hidden_dim
        if len(hidden_dim) != 2:
            raise ValueError("hidden_dim must be an int or a tuple of two ints.")
        return int(hidden_dim[0]), int(hidden_dim[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layers[0](x))
        x = x * self.hidden1_neuron_mask
        x = F.relu(self.layers[1](x))
        x = x * self.hidden2_neuron_mask
        return self.layers[2](x)

    def masked_layers(self) -> list[MaskedLinear]:
        return [module for module in self.modules() if isinstance(module, MaskedLinear)]

    def mask_gradients(self) -> None:
        for layer in self.masked_layers():
            layer.mask_gradients()

    def apply_mask_to_weights(self) -> None:
        self.apply_neuron_masks()
        for layer in self.masked_layers():
            layer.apply_mask_to_weights()

    def prune_and_grow(self, prune_fraction: float, growth_mode: GrowthMode) -> GrowthStats:
        total_stats = MaskedLinear._empty_growth_stats()
        pruned_magnitude_sum = 0.0
        grown_score_sum = 0.0
        for layer in self.masked_layers():
            layer_stats = layer.prune_and_grow(
                prune_fraction=prune_fraction,
                growth_mode=growth_mode,
            )
            for key in ("pruned", "grown"):
                total_stats[key] += layer_stats[key]
            pruned_magnitude_sum += (
                layer_stats["mean_pruned_weight_magnitude"] * layer_stats["pruned"]
            )
            grown_score_sum += layer_stats["mean_grown_score"] * layer_stats["grown"]

        if total_stats["pruned"] > 0:
            total_stats["mean_pruned_weight_magnitude"] = (
                pruned_magnitude_sum / total_stats["pruned"]
            )
        if total_stats["grown"] > 0:
            total_stats["mean_grown_score"] = grown_score_sum / total_stats["grown"]
        return total_stats

    def active_parameter_count(self) -> int:
        return sum(layer.active_parameter_count() for layer in self.masked_layers())

    def hidden_dims(self) -> tuple[int, int]:
        return int(self.hidden1_neuron_mask.numel()), int(self.hidden2_neuron_mask.numel())

    def active_hidden_counts(self) -> tuple[int, int]:
        return (
            int(self.hidden1_neuron_mask.sum().item()),
            int(self.hidden2_neuron_mask.sum().item()),
        )

    def apply_neuron_masks(self) -> None:
        dead_hidden1 = torch.nonzero(self.hidden1_neuron_mask == 0, as_tuple=False).flatten()
        dead_hidden2 = torch.nonzero(self.hidden2_neuron_mask == 0, as_tuple=False).flatten()
        self.layers[0].disable_output_neurons(dead_hidden1)
        self.layers[1].disable_input_neurons(dead_hidden1)
        self.layers[1].disable_output_neurons(dead_hidden2)
        self.layers[2].disable_input_neurons(dead_hidden2)

    def hidden_neuron_importance(
        self,
        hidden_layer_idx: int,
        mode: NeuronPruneMode = "sage",
    ) -> torch.Tensor:
        if hidden_layer_idx == 0:
            producer = self.layers[0]
            consumer = self.layers[1]
            neuron_mask = self.hidden1_neuron_mask.bool()
        elif hidden_layer_idx == 1:
            producer = self.layers[1]
            consumer = self.layers[2]
            neuron_mask = self.hidden2_neuron_mask.bool()
        else:
            raise ValueError("hidden_layer_idx must be 0 or 1.")
        if mode not in ("sage", "magnitude", "random"):
            raise ValueError("mode must be one of: sage, magnitude, random.")

        incoming_weight = (producer.weight.detach().abs() * producer.mask).sum(dim=1)
        outgoing_weight = (consumer.weight.detach().abs() * consumer.mask).sum(dim=0)
        magnitude_score = self._normalize((incoming_weight * outgoing_weight).sqrt())
        if mode == "magnitude":
            return magnitude_score.masked_fill(~neuron_mask, float("-inf"))
        if mode == "random":
            return torch.rand_like(magnitude_score).masked_fill(~neuron_mask, float("-inf"))

        incoming_sage = producer.score_ema.detach().sum(dim=1)
        outgoing_sage = consumer.score_ema.detach().sum(dim=0)
        activity_signal = producer.activation_ema.detach() * producer.grad_output_ema.detach()

        score = (
            self._normalize(activity_signal)
            + magnitude_score
            + self._normalize(incoming_sage + outgoing_sage)
        )
        return score.masked_fill(~neuron_mask, float("-inf"))

    @staticmethod
    def _normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        values = values.float()
        max_value = values.max()
        if max_value <= eps:
            return torch.zeros_like(values)
        return values / (max_value + eps)

    @staticmethod
    def _empty_neuron_stats() -> NeuronStats:
        return {
            "neuron_prune_events": 0.0,
            "pruned_neurons": 0.0,
            "pruned_hidden1": 0.0,
            "pruned_hidden2": 0.0,
            "mean_pruned_neuron_score": 0.0,
            "active_hidden1": 0.0,
            "active_hidden2": 0.0,
        }

    def prune_weak_neurons(
        self,
        prune_fraction: float,
        mode: NeuronPruneMode = "sage",
        protect_fraction: float = 0.05,
        min_hidden_neurons: int = 8,
    ) -> NeuronStats:
        if not 0.0 <= prune_fraction <= 1.0:
            raise ValueError("prune_fraction must be in [0, 1].")
        if not 0.0 <= protect_fraction <= 1.0:
            raise ValueError("protect_fraction must be in [0, 1].")
        if mode not in ("sage", "magnitude", "random"):
            raise ValueError("mode must be one of: sage, magnitude, random.")

        stats = self._empty_neuron_stats()
        pruned_score_sum = 0.0

        for hidden_layer_idx, neuron_mask in (
            (0, self.hidden1_neuron_mask),
            (1, self.hidden2_neuron_mask),
        ):
            alive_idx = torch.nonzero(neuron_mask.bool(), as_tuple=False).flatten()
            alive_count = int(alive_idx.numel())
            max_prunable = max(0, alive_count - min_hidden_neurons)
            prune_count = min(int(alive_count * prune_fraction), max_prunable)
            if prune_count == 0:
                continue

            importance = self.hidden_neuron_importance(hidden_layer_idx, mode=mode)
            candidate_scores = importance.clone()
            protect_count = min(
                int(alive_count * protect_fraction),
                max(0, alive_count - prune_count),
            )
            if protect_count > 0:
                protected_local_idx = torch.topk(
                    importance[alive_idx],
                    protect_count,
                    largest=True,
                ).indices
                protected_idx = alive_idx[protected_local_idx]
                candidate_scores[protected_idx] = float("inf")

            candidate_scores[~neuron_mask.bool()] = float("inf")
            prune_idx = torch.topk(candidate_scores, prune_count, largest=False).indices
            pruned_scores = importance[prune_idx].clamp_min(0.0)
            pruned_score_sum += pruned_scores.sum().item()

            with torch.no_grad():
                neuron_mask[prune_idx] = 0.0

            if hidden_layer_idx == 0:
                stats["pruned_hidden1"] += float(prune_count)
            else:
                stats["pruned_hidden2"] += float(prune_count)
            stats["pruned_neurons"] += float(prune_count)

        if stats["pruned_neurons"] > 0:
            stats["neuron_prune_events"] = 1.0
            stats["mean_pruned_neuron_score"] = pruned_score_sum / stats["pruned_neurons"]

        self.apply_neuron_masks()
        active_hidden1, active_hidden2 = self.active_hidden_counts()
        stats["active_hidden1"] = float(active_hidden1)
        stats["active_hidden2"] = float(active_hidden2)
        return stats

    def compact(self) -> "SparseMLP":
        hidden1_idx = torch.nonzero(self.hidden1_neuron_mask.bool(), as_tuple=False).flatten()
        hidden2_idx = torch.nonzero(self.hidden2_neuron_mask.bool(), as_tuple=False).flatten()
        hidden1_count = int(hidden1_idx.numel())
        hidden2_count = int(hidden2_idx.numel())
        if hidden1_count == 0 or hidden2_count == 0:
            raise ValueError("Cannot compact a model with an empty hidden layer.")

        was_training = self.training
        device = self.layers[0].weight.device
        compacted = SparseMLP(hidden_dim=(hidden1_count, hidden2_count), sparsity=0.0).to(device)

        input_idx = torch.arange(self.layers[0].in_features, device=device)
        output_idx = torch.arange(self.layers[2].out_features, device=device)
        self._copy_layer_subset(self.layers[0], compacted.layers[0], hidden1_idx, input_idx)
        self._copy_layer_subset(self.layers[1], compacted.layers[1], hidden2_idx, hidden1_idx)
        self._copy_layer_subset(self.layers[2], compacted.layers[2], output_idx, hidden2_idx)

        compacted.hidden1_neuron_mask.fill_(1.0)
        compacted.hidden2_neuron_mask.fill_(1.0)
        compacted.train(was_training)
        return compacted

    @staticmethod
    def _copy_layer_subset(
        source: MaskedLinear,
        target: MaskedLinear,
        output_idx: torch.Tensor,
        input_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            target.weight.copy_(source.weight.detach()[output_idx][:, input_idx])
            target.mask.copy_(source.mask.detach()[output_idx][:, input_idx])
            target.score_ema.copy_(source.score_ema.detach()[output_idx][:, input_idx])
            target.activation_ema.copy_(source.activation_ema.detach()[output_idx])
            target.grad_output_ema.copy_(source.grad_output_ema.detach()[output_idx])
            if source.bias is not None and target.bias is not None:
                target.bias.copy_(source.bias.detach()[output_idx])
        target.apply_mask_to_weights()
