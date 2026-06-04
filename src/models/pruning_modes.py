VALID_NEURON_PRUNE_MODES = (
    "sage",
    "sage_pure",
    "sage_mixed",
    "activation",
    "gradient",
    "magnitude",
    "taylor",
    "random",
)


def validate_neuron_prune_mode(mode: str) -> None:
    if mode not in VALID_NEURON_PRUNE_MODES:
        valid = ", ".join(VALID_NEURON_PRUNE_MODES)
        raise ValueError(f"mode must be one of: {valid}.")
