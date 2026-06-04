from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.pruning_modes import validate_neuron_prune_mode
from src.models.sage_conv import MaskedConv2d
from src.models.sage_layer import FocusStats, GrowthMode, GrowthStats, MaskedLinear


NeuronStats = dict[str, float]
NeuronPruneMode = str


class SageCifarCNN(nn.Module):
    """Small CIFAR CNN with SAGE-guided structured channel pruning."""

    conv_output_sizes = (32, 32, 16, 16, 8)

    def __init__(
        self,
        base_channels: int | Sequence[int] = 64,
        num_classes: int = 10,
        sparsity: float = 0.0,
    ) -> None:
        super().__init__()
        channels = self._resolve_channels(base_channels)
        in_channels = (3, *channels[:-1])

        self.convs = nn.ModuleList(
            [
                MaskedConv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    sparsity=sparsity,
                )
                for in_ch, out_ch in zip(in_channels, channels)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_ch) for out_ch in channels])
        self.classifier = MaskedLinear(channels[-1], num_classes, sparsity=sparsity)

        for idx, out_ch in enumerate(channels):
            self.register_buffer(f"channel_mask_{idx}", torch.ones(out_ch))

    @staticmethod
    def _resolve_channels(base_channels: int | Sequence[int]) -> tuple[int, int, int, int, int]:
        if isinstance(base_channels, int):
            return (
                base_channels,
                base_channels,
                base_channels * 2,
                base_channels * 2,
                base_channels * 4,
            )
        channels = tuple(int(value) for value in base_channels)
        if len(channels) != 5:
            raise ValueError("base_channels must be an int or a sequence of five channel counts.")
        return channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_block(x, 0)
        x = self._conv_block(x, 1)
        x = F.max_pool2d(x, 2)
        x = self._conv_block(x, 2)
        x = self._conv_block(x, 3)
        x = F.max_pool2d(x, 2)
        x = self._conv_block(x, 4)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = x * self.channel_masks()[-1]
        return self.classifier(x)

    def _conv_block(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x = self.convs[layer_idx](x)
        x = self.bns[layer_idx](x)
        x = F.relu(x)
        return x * self.channel_masks()[layer_idx].view(1, -1, 1, 1)

    def channel_masks(self) -> list[torch.Tensor]:
        return [getattr(self, f"channel_mask_{idx}") for idx in range(len(self.convs))]

    def masked_layers(self) -> list[nn.Module]:
        return [*self.convs, self.classifier]

    def mask_gradients(self) -> None:
        for layer in self.masked_layers():
            layer.mask_gradients()

    def apply_mask_to_weights(self) -> None:
        self.apply_channel_masks()
        for layer in self.masked_layers():
            layer.apply_mask_to_weights()

    def prune_and_grow(self, prune_fraction: float, growth_mode: GrowthMode) -> GrowthStats:
        total_stats = self._empty_growth_stats()
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

    @staticmethod
    def _empty_growth_stats() -> GrowthStats:
        return {
            "pruned": 0.0,
            "grown": 0.0,
            "mean_pruned_weight_magnitude": 0.0,
            "mean_grown_score": 0.0,
        }

    def active_parameter_count(self) -> int:
        return sum(layer.active_parameter_count() for layer in self.masked_layers())

    def forward_flop_metrics(self) -> dict[str, float]:
        physical_flops = 0.0
        active_flops = 0.0
        for conv, output_size in zip(self.convs, self.conv_output_sizes):
            output_positions = output_size * output_size
            physical_flops += 2.0 * conv.weight.numel() * output_positions
            active_flops += 2.0 * conv.mask.sum().item() * output_positions

        physical_flops += 2.0 * self.classifier.weight.numel()
        active_flops += 2.0 * self.classifier.mask.sum().item()
        return {
            "physical_forward_flops": physical_flops,
            "active_forward_flops": active_flops,
        }

    def apply_sage_gradient_focus(
        self,
        boost_factor: float,
        boost_fraction: float,
        weak_grad_decay: float = 1.0,
    ) -> FocusStats:
        total_stats = self._empty_focus_stats()
        boosted_score_sum = 0.0
        for layer in self.masked_layers():
            layer_stats = layer.apply_sage_gradient_focus(
                boost_factor=boost_factor,
                boost_fraction=boost_fraction,
                weak_grad_decay=weak_grad_decay,
            )
            total_stats["boosted_edges"] += layer_stats["boosted_edges"]
            total_stats["weak_scaled_edges"] += layer_stats["weak_scaled_edges"]
            boosted_score_sum += (
                layer_stats["mean_boosted_score"] * layer_stats["boosted_edges"]
            )

        if total_stats["boosted_edges"] > 0:
            total_stats["mean_boosted_score"] = (
                boosted_score_sum / total_stats["boosted_edges"]
            )
        return total_stats

    @staticmethod
    def _empty_focus_stats() -> FocusStats:
        return {
            "boosted_edges": 0.0,
            "weak_scaled_edges": 0.0,
            "mean_boosted_score": 0.0,
        }

    def channel_dims(self) -> tuple[int, ...]:
        return tuple(int(mask.numel()) for mask in self.channel_masks())

    def active_channel_counts(self) -> tuple[int, ...]:
        return tuple(int(mask.sum().item()) for mask in self.channel_masks())

    def structure_metrics(self) -> dict[str, float]:
        metrics = {}
        for idx, (dim, active) in enumerate(zip(self.channel_dims(), self.active_channel_counts())):
            metrics[f"conv{idx}_channels"] = float(dim)
            metrics[f"active_conv{idx}_channels"] = float(active)
        metrics["total_active_channels"] = float(sum(self.active_channel_counts()))
        return metrics

    def apply_channel_masks(self) -> None:
        masks = self.channel_masks()
        for idx, channel_mask in enumerate(masks):
            dead_idx = torch.nonzero(channel_mask == 0, as_tuple=False).flatten()
            self.convs[idx].disable_output_channels(dead_idx)
            self._disable_bn_channels(self.bns[idx], dead_idx)
            if idx + 1 < len(self.convs):
                self.convs[idx + 1].disable_input_channels(dead_idx)
            else:
                self.classifier.disable_input_neurons(dead_idx)

    @staticmethod
    def _disable_bn_channels(batch_norm: nn.BatchNorm2d, channel_idx: torch.Tensor) -> None:
        if channel_idx.numel() == 0:
            return
        with torch.no_grad():
            if batch_norm.affine:
                batch_norm.weight[channel_idx] = 0.0
                batch_norm.bias[channel_idx] = 0.0
                if batch_norm.weight.grad is not None:
                    batch_norm.weight.grad[channel_idx] = 0.0
                if batch_norm.bias.grad is not None:
                    batch_norm.bias.grad[channel_idx] = 0.0
            batch_norm.running_mean[channel_idx] = 0.0
            batch_norm.running_var[channel_idx] = 1.0

    def channel_importance(
        self,
        conv_idx: int,
        mode: NeuronPruneMode = "sage",
    ) -> torch.Tensor:
        if not 0 <= conv_idx < len(self.convs):
            raise ValueError("conv_idx is out of range.")
        validate_neuron_prune_mode(mode)

        producer = self.convs[conv_idx]
        channel_mask = self.channel_masks()[conv_idx].bool()

        incoming_weight = (producer.weight.detach().abs() * producer.mask).sum(dim=(1, 2, 3))
        if conv_idx + 1 < len(self.convs):
            consumer = self.convs[conv_idx + 1]
            outgoing_weight = (consumer.weight.detach().abs() * consumer.mask).sum(dim=(0, 2, 3))
            outgoing_sage = consumer.score_ema.detach().sum(dim=(0, 2, 3))
        else:
            outgoing_weight = (
                self.classifier.weight.detach().abs() * self.classifier.mask
            ).sum(dim=0)
            outgoing_sage = self.classifier.score_ema.detach().sum(dim=0)

        magnitude_score = self._normalize((incoming_weight * outgoing_weight).sqrt())
        if mode == "magnitude":
            return magnitude_score.masked_fill(~channel_mask, float("-inf"))
        if mode == "random":
            return torch.rand_like(magnitude_score).masked_fill(~channel_mask, float("-inf"))
        if mode == "activation":
            return self._normalize(producer.activation_ema.detach()).masked_fill(
                ~channel_mask,
                float("-inf"),
            )
        if mode == "gradient":
            return self._normalize(producer.grad_output_ema.detach()).masked_fill(
                ~channel_mask,
                float("-inf"),
            )

        incoming_sage = producer.score_ema.detach().sum(dim=(1, 2, 3))
        activity_signal = producer.activation_ema.detach() * producer.grad_output_ema.detach()
        if mode == "sage_pure":
            return self._normalize(activity_signal).masked_fill(~channel_mask, float("-inf"))
        if mode == "taylor":
            incoming_taylor = self._conv_taylor_flow(producer, dim=1)
            if conv_idx + 1 < len(self.convs):
                outgoing_taylor = self._conv_taylor_flow(self.convs[conv_idx + 1], dim=0)
            else:
                outgoing_taylor = self._linear_taylor_flow(self.classifier, dim=0)
            taylor_score = self._normalize((incoming_taylor * outgoing_taylor).sqrt())
            return taylor_score.masked_fill(~channel_mask, float("-inf"))

        score = (
            self._normalize(activity_signal)
            + magnitude_score
            + self._normalize(incoming_sage + outgoing_sage)
        )
        return score.masked_fill(~channel_mask, float("-inf"))

    @staticmethod
    def _conv_taylor_flow(layer: MaskedConv2d, dim: int) -> torch.Tensor:
        if layer.weight.grad is None:
            output_size = layer.out_channels if dim == 1 else layer.in_channels
            return torch.zeros(output_size, device=layer.weight.device, dtype=layer.weight.dtype)
        dims = (1, 2, 3) if dim == 1 else (0, 2, 3)
        return (layer.weight.detach() * layer.weight.grad.detach()).abs().mul(layer.mask).sum(dim=dims)

    @staticmethod
    def _linear_taylor_flow(layer: MaskedLinear, dim: int) -> torch.Tensor:
        if layer.weight.grad is None:
            output_size = layer.weight.shape[1 - dim]
            return torch.zeros(output_size, device=layer.weight.device, dtype=layer.weight.dtype)
        return (layer.weight.detach() * layer.weight.grad.detach()).abs().mul(layer.mask).sum(dim=dim)

    @staticmethod
    def _normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        values = values.float()
        max_value = values.max()
        if max_value <= eps:
            return torch.zeros_like(values)
        return values / (max_value + eps)

    @staticmethod
    def _empty_neuron_stats() -> NeuronStats:
        stats = {
            "neuron_prune_events": 0.0,
            "pruned_neurons": 0.0,
            "pruned_hidden1": 0.0,
            "pruned_hidden2": 0.0,
            "mean_pruned_neuron_score": 0.0,
            "active_hidden1": 0.0,
            "active_hidden2": 0.0,
            "pruned_channels": 0.0,
            "mean_pruned_channel_score": 0.0,
        }
        for idx in range(5):
            stats[f"pruned_conv{idx}_channels"] = 0.0
            stats[f"active_conv{idx}_channels"] = 0.0
        return stats

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
        validate_neuron_prune_mode(mode)

        stats = self._empty_neuron_stats()
        pruned_score_sum = 0.0

        for conv_idx, channel_mask in enumerate(self.channel_masks()):
            alive_idx = torch.nonzero(channel_mask.bool(), as_tuple=False).flatten()
            alive_count = int(alive_idx.numel())
            max_prunable = max(0, alive_count - min_hidden_neurons)
            prune_count = min(int(alive_count * prune_fraction), max_prunable)
            if prune_count == 0:
                continue

            importance = self.channel_importance(conv_idx, mode=mode)
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

            candidate_scores[~channel_mask.bool()] = float("inf")
            prune_idx = torch.topk(candidate_scores, prune_count, largest=False).indices
            pruned_scores = importance[prune_idx].clamp_min(0.0)
            pruned_score_sum += pruned_scores.sum().item()

            with torch.no_grad():
                channel_mask[prune_idx] = 0.0

            stats[f"pruned_conv{conv_idx}_channels"] += float(prune_count)
            stats["pruned_channels"] += float(prune_count)
            stats["pruned_neurons"] += float(prune_count)

        if stats["pruned_channels"] > 0:
            stats["neuron_prune_events"] = 1.0
            stats["mean_pruned_channel_score"] = pruned_score_sum / stats["pruned_channels"]
            stats["mean_pruned_neuron_score"] = stats["mean_pruned_channel_score"]

        self.apply_channel_masks()
        for idx, active_count in enumerate(self.active_channel_counts()):
            stats[f"active_conv{idx}_channels"] = float(active_count)
        return stats

    def compact(self) -> "SageCifarCNN":
        active_idx = [
            torch.nonzero(mask.bool(), as_tuple=False).flatten()
            for mask in self.channel_masks()
        ]
        if any(idx.numel() == 0 for idx in active_idx):
            raise ValueError("Cannot compact a model with an empty conv layer.")

        was_training = self.training
        device = self.convs[0].weight.device
        compacted_channels = tuple(int(idx.numel()) for idx in active_idx)
        compacted = SageCifarCNN(
            base_channels=compacted_channels,
            num_classes=self.classifier.out_features,
            sparsity=0.0,
        ).to(device)

        input_idx = torch.arange(3, device=device)
        for conv_idx, output_idx in enumerate(active_idx):
            conv_input_idx = input_idx if conv_idx == 0 else active_idx[conv_idx - 1]
            self._copy_conv_subset(
                self.convs[conv_idx],
                compacted.convs[conv_idx],
                output_idx,
                conv_input_idx,
            )
            self._copy_bn_subset(self.bns[conv_idx], compacted.bns[conv_idx], output_idx)
            compacted.channel_masks()[conv_idx].fill_(1.0)

        output_idx = torch.arange(self.classifier.out_features, device=device)
        self._copy_linear_subset(
            self.classifier,
            compacted.classifier,
            output_idx,
            active_idx[-1],
        )

        compacted.train(was_training)
        return compacted

    @staticmethod
    def _copy_conv_subset(
        source: MaskedConv2d,
        target: MaskedConv2d,
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

    @staticmethod
    def _copy_bn_subset(
        source: nn.BatchNorm2d,
        target: nn.BatchNorm2d,
        channel_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if source.affine and target.affine:
                target.weight.copy_(source.weight.detach()[channel_idx])
                target.bias.copy_(source.bias.detach()[channel_idx])
            target.running_mean.copy_(source.running_mean.detach()[channel_idx])
            target.running_var.copy_(source.running_var.detach()[channel_idx])
            target.num_batches_tracked.copy_(source.num_batches_tracked.detach())

    @staticmethod
    def _copy_linear_subset(
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
