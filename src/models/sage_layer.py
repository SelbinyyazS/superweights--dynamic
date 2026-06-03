import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


GrowthMode = Literal["random", "gradient", "sage"]
GrowthStats = dict[str, float]


class _MaskedWeight(torch.autograd.Function):
    """Masked forward with dense weight gradients for growth scoring."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class MaskedLinear(nn.Module):
    """Linear layer with dense weights and a binary edge mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.95,
        bias: bool = False,
        score_ema_decay: float = 0.9,
        grow_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1).")
        if not 0.0 <= score_ema_decay < 1.0:
            raise ValueError("score_ema_decay must be in [0, 1).")

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.score_ema_decay = score_ema_decay
        self.grow_init_std = grow_init_std

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        mask = self._initial_mask(out_features, in_features, sparsity)
        self.register_buffer("mask", mask)
        self.register_buffer("score_ema", torch.zeros(out_features, in_features))
        self.register_buffer("activation_ema", torch.zeros(out_features))
        self.register_buffer("grad_output_ema", torch.zeros(out_features))
        self.last_input_activation: torch.Tensor | None = None
        self.last_growth_stats = self._empty_growth_stats()

        self.reset_parameters()

    @staticmethod
    def _empty_growth_stats() -> GrowthStats:
        return {
            "pruned": 0.0,
            "grown": 0.0,
            "mean_pruned_weight_magnitude": 0.0,
            "mean_grown_score": 0.0,
        }

    @staticmethod
    def _initial_mask(out_features: int, in_features: int, sparsity: float) -> torch.Tensor:
        total = out_features * in_features
        active = max(1, int(round(total * (1.0 - sparsity))))
        flat_mask = torch.zeros(total)
        flat_mask[torch.randperm(total)[:active]] = 1.0
        return flat_mask.view(out_features, in_features)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.apply_mask_to_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input_activation = x.detach()
        masked_weight = _MaskedWeight.apply(self.weight, self.mask)
        output = F.linear(x, masked_weight, self.bias)
        if self.training:
            with torch.no_grad():
                output_batch = self._as_batch(output.detach())
                output_score = output_batch.abs().mean(dim=0)
                self.activation_ema.mul_(self.score_ema_decay)
                self.activation_ema.add_(output_score, alpha=1.0 - self.score_ema_decay)
        if self.training and output.requires_grad:
            output.register_hook(self._make_score_hook(x.detach()))
        return output

    def _make_score_hook(self, input_activation: torch.Tensor):
        def hook(grad_output: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                input_batch = self._as_batch(input_activation)
                grad_batch = self._as_batch(grad_output.detach())
                input_score = input_batch.abs().mean(dim=0)
                grad_score = grad_batch.abs().mean(dim=0)
                batch_score = grad_score.unsqueeze(1) * input_score.unsqueeze(0)
                self.score_ema.mul_(self.score_ema_decay)
                self.score_ema.add_(batch_score, alpha=1.0 - self.score_ema_decay)
                self.grad_output_ema.mul_(self.score_ema_decay)
                self.grad_output_ema.add_(grad_score, alpha=1.0 - self.score_ema_decay)
            return grad_output

        return hook

    @staticmethod
    def _as_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        if tensor.dim() == 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])

    def mask_gradients(self) -> None:
        if self.weight.grad is not None:
            self.weight.grad.mul_(self.mask)

    def apply_mask_to_weights(self) -> None:
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def active_parameter_count(self) -> int:
        return int(self.mask.sum().item())

    def disable_output_neurons(self, neuron_idx: torch.Tensor) -> None:
        if neuron_idx.numel() == 0:
            return
        with torch.no_grad():
            self.mask[neuron_idx, :] = 0.0
            self.weight[neuron_idx, :] = 0.0
            self.score_ema[neuron_idx, :] = 0.0
            self.activation_ema[neuron_idx] = 0.0
            self.grad_output_ema[neuron_idx] = 0.0
            if self.weight.grad is not None:
                self.weight.grad[neuron_idx, :] = 0.0
            if self.bias is not None:
                self.bias[neuron_idx] = 0.0
                if self.bias.grad is not None:
                    self.bias.grad[neuron_idx] = 0.0

    def disable_input_neurons(self, neuron_idx: torch.Tensor) -> None:
        if neuron_idx.numel() == 0:
            return
        with torch.no_grad():
            self.mask[:, neuron_idx] = 0.0
            self.weight[:, neuron_idx] = 0.0
            self.score_ema[:, neuron_idx] = 0.0
            if self.weight.grad is not None:
                self.weight.grad[:, neuron_idx] = 0.0

    def prune_and_grow(self, prune_fraction: float, growth_mode: GrowthMode) -> GrowthStats:
        if not 0.0 <= prune_fraction <= 1.0:
            raise ValueError("prune_fraction must be in [0, 1].")
        if growth_mode not in ("random", "gradient", "sage"):
            raise ValueError("growth_mode must be one of: random, gradient, sage.")

        stats = self._empty_growth_stats()
        active_mask = self.mask.bool()
        active_count = int(active_mask.sum().item())
        prune_count = int(active_count * prune_fraction)
        if prune_count == 0:
            self.last_growth_stats = stats
            return stats

        with torch.no_grad():
            flat_mask = self.mask.view(-1)
            flat_weight = self.weight.view(-1)
            flat_score_ema = self.score_ema.view(-1)

            active_scores = self.weight.detach().abs().masked_fill(~active_mask, float("inf"))
            prune_idx = torch.topk(active_scores.view(-1), prune_count, largest=False).indices
            stats["pruned"] = float(prune_count)
            stats["mean_pruned_weight_magnitude"] = flat_weight[prune_idx].abs().mean().item()
            flat_mask[prune_idx] = 0.0
            flat_weight[prune_idx] = 0.0
            flat_score_ema[prune_idx] = 0.0

            grow_scores = self._growth_scores(growth_mode)
            inactive_mask = ~self.mask.bool()
            grow_count = min(prune_count, int(inactive_mask.sum().item()))
            if grow_count == 0:
                self.last_growth_stats = stats
                return stats

            grow_scores = grow_scores.masked_fill(~inactive_mask, float("-inf"))
            grow_idx = torch.topk(grow_scores.view(-1), grow_count, largest=True).indices
            stats["grown"] = float(grow_count)
            stats["mean_grown_score"] = grow_scores.view(-1)[grow_idx].mean().item()

            flat_mask[grow_idx] = 1.0
            flat_weight[grow_idx] = torch.randn(
                grow_count,
                device=self.weight.device,
                dtype=self.weight.dtype,
            ) * self.grow_init_std
            flat_score_ema[grow_idx] = 0.0

        self.last_growth_stats = stats
        return stats

    def _growth_scores(self, growth_mode: GrowthMode) -> torch.Tensor:
        if growth_mode == "random":
            return torch.rand_like(self.weight)
        if growth_mode == "gradient":
            if self.weight.grad is None:
                return torch.zeros_like(self.weight)
            return self.weight.grad.detach().abs()
        return self.score_ema.detach()
