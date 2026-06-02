import torch
from torch import nn
from torch.nn import functional as F

from src.models.sage_layer import GrowthMode, MaskedLinear


class SparseMLP(nn.Module):
    """Three-layer sparse MLP for flattened 28x28 images."""

    def __init__(self, hidden_dim: int = 256, sparsity: float = 0.95) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MaskedLinear(784, hidden_dim, sparsity=sparsity),
                MaskedLinear(hidden_dim, hidden_dim, sparsity=sparsity),
                MaskedLinear(hidden_dim, 10, sparsity=sparsity),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        return self.layers[2](x)

    def masked_layers(self) -> list[MaskedLinear]:
        return [module for module in self.modules() if isinstance(module, MaskedLinear)]

    def mask_gradients(self) -> None:
        for layer in self.masked_layers():
            layer.mask_gradients()

    def apply_mask_to_weights(self) -> None:
        for layer in self.masked_layers():
            layer.apply_mask_to_weights()

    def prune_and_grow(self, prune_fraction: float, growth_mode: GrowthMode) -> int:
        return sum(
            layer.prune_and_grow(prune_fraction=prune_fraction, growth_mode=growth_mode)
            for layer in self.masked_layers()
        )

    def active_parameter_count(self) -> int:
        return sum(layer.active_parameter_count() for layer in self.masked_layers())
