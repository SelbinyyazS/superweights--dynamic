import torch
from torch.nn import functional as F

from src.models.sparse_mlp import SparseMLP
from src.utils.metrics import active_parameter_count, superweight_concentration


def main() -> None:
    torch.manual_seed(7)
    model = SparseMLP(hidden_dim=32, sparsity=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    before_active = active_parameter_count(model)
    inputs = torch.randn(16, 1, 28, 28)
    targets = torch.randint(0, 10, (16,))

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    sage_score_sum = sum(layer.score_ema.sum().item() for layer in model.masked_layers())
    assert sage_score_sum > 0.0, "SAGE score EMA did not update during backward."

    growth_stats = model.prune_and_grow(prune_fraction=0.1, growth_mode="sage")
    model.mask_gradients()

    for layer in model.masked_layers():
        inactive_grad = layer.weight.grad[layer.mask == 0]
        if inactive_grad.numel() > 0:
            assert inactive_grad.abs().max().item() == 0.0

    optimizer.step()
    model.apply_mask_to_weights()

    after_active = active_parameter_count(model)
    assert before_active == after_active, (before_active, after_active)

    for layer in model.masked_layers():
        inactive_weight = layer.weight.detach()[layer.mask == 0]
        if inactive_weight.numel() > 0:
            assert inactive_weight.abs().max().item() == 0.0

    print(
        "smoke ok "
        f"loss={loss.item():.4f} "
        f"active={after_active} "
        f"grown={int(growth_stats['grown'])} "
        f"concentration={superweight_concentration(model):.4f}"
    )

    before_neuron_prune_active = active_parameter_count(model)
    neuron_stats = model.prune_weak_neurons(
        prune_fraction=0.25,
        protect_fraction=0.10,
        min_hidden_neurons=8,
    )
    after_neuron_prune_active = active_parameter_count(model)
    assert neuron_stats["pruned_neurons"] > 0.0
    assert after_neuron_prune_active < before_neuron_prune_active

    hidden_counts = model.active_hidden_counts()
    compacted = model.compact()
    assert compacted.hidden_dims() == hidden_counts
    assert compacted.active_hidden_counts() == hidden_counts

    model.eval()
    compacted.eval()
    with torch.no_grad():
        original_logits = model(inputs)
        compacted_logits = compacted(inputs)
    assert torch.allclose(original_logits, compacted_logits, atol=1e-5)

    print(
        "neuron prune ok "
        f"pruned={int(neuron_stats['pruned_neurons'])} "
        f"hidden={hidden_counts} "
        f"active={after_neuron_prune_active} "
        f"compacted_dims={compacted.hidden_dims()}"
    )


if __name__ == "__main__":
    main()
