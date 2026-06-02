import torch
from torch import nn

from src.models.sage_layer import MaskedLinear


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def active_parameter_count(model: nn.Module) -> int:
    return sum(
        int(module.mask.sum().item())
        for module in model.modules()
        if isinstance(module, MaskedLinear)
    )


def superweight_concentration(model: nn.Module, top_fraction: float = 0.01) -> float:
    """Share of active weight magnitude held by the top active weights."""

    active_weights = []
    for module in model.modules():
        if isinstance(module, MaskedLinear):
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
