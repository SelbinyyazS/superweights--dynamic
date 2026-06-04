import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.sage_layer import FocusStats, GrowthMode, GrowthStats, _MaskedWeight


class MaskedConv2d(nn.Module):
    """Conv2d with dense weights, a binary weight mask, and SAGE score tracking."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
        sparsity: float = 0.0,
        score_ema_decay: float = 0.9,
        grow_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1).")
        if not 0.0 <= score_ema_decay < 1.0:
            raise ValueError("score_ema_decay must be in [0, 1).")

        kernel_size = self._pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = self._pair(dilation)
        self.sparsity = sparsity
        self.score_ema_decay = score_ema_decay
        self.grow_init_std = grow_init_std

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", self._initial_mask(self.weight.shape, sparsity))
        self.register_buffer("score_ema", torch.zeros_like(self.weight))
        self.register_buffer("activation_ema", torch.zeros(out_channels))
        self.register_buffer("grad_output_ema", torch.zeros(out_channels))
        self.last_input_activation: torch.Tensor | None = None
        self.last_growth_stats = self._empty_growth_stats()

        self.reset_parameters()

    @staticmethod
    def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, tuple):
            return value
        return (value, value)

    @staticmethod
    def _empty_growth_stats() -> GrowthStats:
        return {
            "pruned": 0.0,
            "grown": 0.0,
            "mean_pruned_weight_magnitude": 0.0,
            "mean_grown_score": 0.0,
        }

    @staticmethod
    def _empty_focus_stats() -> FocusStats:
        return {
            "boosted_edges": 0.0,
            "weak_scaled_edges": 0.0,
            "mean_boosted_score": 0.0,
        }

    @staticmethod
    def _initial_mask(shape: torch.Size, sparsity: float) -> torch.Tensor:
        total = math.prod(shape)
        active = max(1, int(round(total * (1.0 - sparsity))))
        flat_mask = torch.zeros(total)
        flat_mask[torch.randperm(total)[:active]] = 1.0
        return flat_mask.view(shape)

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
        output = F.conv2d(
            x,
            masked_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.training:
            with torch.no_grad():
                output_score = self._channel_mean_abs(output.detach())
                self.activation_ema.mul_(self.score_ema_decay)
                self.activation_ema.add_(output_score, alpha=1.0 - self.score_ema_decay)
        if self.training and output.requires_grad:
            output.register_hook(self._make_score_hook(x.detach()))
        return output

    def _make_score_hook(self, input_activation: torch.Tensor):
        def hook(grad_output: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                input_score = self._channel_mean_abs(input_activation)
                grad_score = self._channel_mean_abs(grad_output.detach())
                channel_score = grad_score.unsqueeze(1) * input_score.unsqueeze(0)
                batch_score = channel_score[:, :, None, None].expand_as(self.score_ema)
                self.score_ema.mul_(self.score_ema_decay)
                self.score_ema.add_(batch_score, alpha=1.0 - self.score_ema_decay)
                self.grad_output_ema.mul_(self.score_ema_decay)
                self.grad_output_ema.add_(grad_score, alpha=1.0 - self.score_ema_decay)
            return grad_output

        return hook

    @staticmethod
    def _channel_mean_abs(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.abs().mean(dim=0)
        if tensor.dim() == 4:
            return tensor.abs().mean(dim=(0, 2, 3))
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1).abs().mean(dim=(0, 2))

    def mask_gradients(self) -> None:
        if self.weight.grad is not None:
            self.weight.grad.mul_(self.mask)

    def apply_mask_to_weights(self) -> None:
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def active_parameter_count(self) -> int:
        return int(self.mask.sum().item())

    def apply_sage_gradient_focus(
        self,
        boost_factor: float,
        boost_fraction: float,
        weak_grad_decay: float = 1.0,
    ) -> FocusStats:
        stats = self._empty_focus_stats()
        if self.weight.grad is None or boost_fraction <= 0.0:
            return stats
        if boost_factor == 1.0 and weak_grad_decay == 1.0:
            return stats
        if boost_factor < 0.0:
            raise ValueError("boost_factor must be non-negative.")
        if not 0.0 <= boost_fraction <= 1.0:
            raise ValueError("boost_fraction must be in [0, 1].")
        if weak_grad_decay < 0.0:
            raise ValueError("weak_grad_decay must be non-negative.")

        active_mask = self.mask.bool()
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            return stats

        boost_count = max(1, int(active_count * boost_fraction))
        boost_count = min(boost_count, active_count)

        with torch.no_grad():
            flat_grad = self.weight.grad.view(-1)
            flat_active = active_mask.view(-1)
            flat_scores = self.score_ema.detach().view(-1)
            candidate_scores = flat_scores.masked_fill(~flat_active, float("-inf"))
            boost_idx = torch.topk(candidate_scores, boost_count, largest=True).indices

            weak_mask = flat_active.clone()
            weak_mask[boost_idx] = False
            if weak_grad_decay != 1.0:
                flat_grad[weak_mask] *= weak_grad_decay
                stats["weak_scaled_edges"] = float(int(weak_mask.sum().item()))

            flat_grad[boost_idx] *= boost_factor
            stats["boosted_edges"] = float(boost_count)
            stats["mean_boosted_score"] = candidate_scores[boost_idx].mean().item()
        return stats

    def disable_output_channels(self, channel_idx: torch.Tensor) -> None:
        if channel_idx.numel() == 0:
            return
        with torch.no_grad():
            self.mask[channel_idx, :, :, :] = 0.0
            self.weight[channel_idx, :, :, :] = 0.0
            self.score_ema[channel_idx, :, :, :] = 0.0
            self.activation_ema[channel_idx] = 0.0
            self.grad_output_ema[channel_idx] = 0.0
            if self.weight.grad is not None:
                self.weight.grad[channel_idx, :, :, :] = 0.0
            if self.bias is not None:
                self.bias[channel_idx] = 0.0
                if self.bias.grad is not None:
                    self.bias.grad[channel_idx] = 0.0

    def disable_input_channels(self, channel_idx: torch.Tensor) -> None:
        if channel_idx.numel() == 0:
            return
        with torch.no_grad():
            self.mask[:, channel_idx, :, :] = 0.0
            self.weight[:, channel_idx, :, :] = 0.0
            self.score_ema[:, channel_idx, :, :] = 0.0
            if self.weight.grad is not None:
                self.weight.grad[:, channel_idx, :, :] = 0.0

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
