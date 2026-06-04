import torch
from torch import nn

from src.models.sage_conv import MaskedConv2d
from src.models.sage_layer import MaskedLinear


MASKED_LAYER_TYPES = (MaskedConv2d, MaskedLinear)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def masked_layers(model: nn.Module) -> list[nn.Module]:
    return [module for module in model.modules() if isinstance(module, MASKED_LAYER_TYPES)]


def active_parameter_count(model: nn.Module) -> int:
    return sum(int(module.mask.sum().item()) for module in masked_layers(model))


def physical_parameter_count(model: nn.Module) -> int:
    return sum(int(module.weight.numel()) for module in masked_layers(model))


def superweight_concentration(model: nn.Module, top_fraction: float = 0.01) -> float:
    """Share of active weight magnitude held by the top active weights."""

    active_weights = []
    for module in masked_layers(model):
        weights = module.weight.detach().abs()[module.mask.bool()]
        if weights.numel() > 0:
            active_weights.append(weights.flatten())

    if not active_weights:
        return 0.0

    weights = torch.cat(active_weights)
    total = weights.sum()
    if total.item() == 0.0:
        return 0.0

    top_k = max(1, int(weights.numel() * top_fraction))
    return (torch.topk(weights, top_k).values.sum() / total).item()


def layerwise_metrics(model: nn.Module) -> dict[str, float]:
    metrics = {}
    layers = masked_layers(model)
    for layer_idx, layer in enumerate(layers):
        active_params = layer.active_parameter_count()
        total_params = layer.mask.numel()
        active_scores = layer.score_ema.detach()[layer.mask.bool()]
        inactive_scores = layer.score_ema.detach()[~layer.mask.bool()]

        metrics[f"layer_{layer_idx}_active_parameter_count"] = float(active_params)
        metrics[f"layer_{layer_idx}_density"] = active_params / total_params
        metrics[f"layer_{layer_idx}_active_score_mean"] = (
            active_scores.mean().item() if active_scores.numel() > 0 else 0.0
        )
        metrics[f"layer_{layer_idx}_inactive_score_mean"] = (
            inactive_scores.mean().item() if inactive_scores.numel() > 0 else 0.0
        )
        metrics[f"layer_{layer_idx}_score_max"] = layer.score_ema.max().item()
        metrics[f"layer_{layer_idx}_last_pruned"] = layer.last_growth_stats["pruned"]
        metrics[f"layer_{layer_idx}_last_grown"] = layer.last_growth_stats["grown"]
    return metrics


def structure_metrics(model: nn.Module) -> dict[str, float]:
    metrics = {
        "physical_parameter_count": float(physical_parameter_count(model)),
    }
    if hasattr(model, "forward_flop_metrics"):
        metrics.update(model.forward_flop_metrics())
    if hasattr(model, "structure_metrics"):
        metrics.update(model.structure_metrics())
    if hasattr(model, "hidden_dims") and hasattr(model, "active_hidden_counts"):
        hidden1_dim, hidden2_dim = model.hidden_dims()
        active_hidden1, active_hidden2 = model.active_hidden_counts()
        metrics.update(
            {
                "hidden1_dim": float(hidden1_dim),
                "hidden2_dim": float(hidden2_dim),
                "active_hidden1": float(active_hidden1),
                "active_hidden2": float(active_hidden2),
            }
        )
    return metrics
