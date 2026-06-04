import argparse
import csv
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.sage_cnn import SageCifarCNN
from src.models.sparse_mlp import SparseMLP
from src.utils.metrics import (
    active_parameter_count,
    layerwise_metrics,
    structure_metrics,
    superweight_concentration,
)


BASE_FIELDNAMES = [
    "epoch",
    "loss",
    "accuracy",
    "active_parameter_count",
    "superweight_concentration",
    "growth_events",
    "pruned_edges",
    "grown_edges",
    "mean_pruned_weight_magnitude",
    "mean_grown_score",
    "neuron_prune_events",
    "neuron_prune_mode",
    "pruned_neurons",
    "pruned_hidden1",
    "pruned_hidden2",
    "mean_pruned_neuron_score",
    "pruned_channels",
    "pruned_conv0_channels",
    "pruned_conv1_channels",
    "pruned_conv2_channels",
    "pruned_conv3_channels",
    "pruned_conv4_channels",
    "mean_pruned_channel_score",
    "compacted",
    "sage_focus_steps",
    "mean_boosted_edges",
    "mean_weak_scaled_edges",
    "mean_boosted_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SAGE sparse network prototype.")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion-mnist", "fashion_mnist", "cifar10", "cifar-10"],
        default="mnist",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "mlp", "cifar_cnn"],
        default="auto",
        help="Model family. auto uses cifar_cnn for CIFAR-10 and mlp otherwise.",
    )
    parser.add_argument("--sparsity", type=float, default=0.95)
    parser.add_argument("--dense_start", action="store_true", help="Start with all edges active.")
    parser.add_argument("--growth_mode", choices=["random", "gradient", "sage"], default="sage")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--growth_interval", type=int, default=100)
    parser.add_argument("--prune_fraction", type=float, default=0.05)
    parser.add_argument("--neuron_prune_start_epoch", type=int, default=0)
    parser.add_argument("--neuron_prune_end_epoch", type=int, default=0)
    parser.add_argument("--neuron_prune_interval", type=int, default=1)
    parser.add_argument("--neuron_prune_fraction", type=float, default=0.0)
    parser.add_argument(
        "--neuron_prune_mode",
        choices=["sage", "magnitude", "random"],
        default="sage",
    )
    parser.add_argument("--neuron_protect_fraction", type=float, default=0.05)
    parser.add_argument("--min_hidden_neurons", type=int, default=8)
    parser.add_argument(
        "--compact_epoch",
        type=int,
        default=0,
        help="Epoch for physical compaction; -1 compacts after the final training epoch, 0 disables it.",
    )
    parser.add_argument("--post_compact_epochs", type=int, default=0)
    parser.add_argument("--sage_focus_start_epoch", type=int, default=0)
    parser.add_argument("--sage_focus_end_epoch", type=int, default=0)
    parser.add_argument("--sage_grad_boost", type=float, default=1.0)
    parser.add_argument("--sage_boost_fraction", type=float, default=0.0)
    parser.add_argument("--weak_grad_decay", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/train.csv"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_batches", type=int, default=0, help="Limit train batches per epoch; 0 means all.")
    parser.add_argument("--eval_batches", type=int, default=0, help="Limit eval batches; 0 means all.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalized_dataset_name(dataset: str) -> str:
    dataset_name = dataset.replace("_", "-").lower()
    if dataset_name == "cifar-10":
        return "cifar10"
    return dataset_name


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    dataset_name = normalized_dataset_name(args.dataset)
    if dataset_name in ("mnist", "fashion-mnist"):
        dataset_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = dataset_cls(args.data_dir, train=True, download=True, transform=transform)
        eval_dataset = dataset_cls(args.data_dir, train=False, download=True, transform=transform)
    elif dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        eval_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=eval_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    if normalized_dataset_name(args.dataset) == "cifar10":
        return "cifar_cnn"
    return "mlp"


def build_model(args: argparse.Namespace, sparsity: float) -> torch.nn.Module:
    model_name = resolve_model_name(args)
    dataset_name = normalized_dataset_name(args.dataset)
    if model_name == "mlp":
        if dataset_name not in ("mnist", "fashion-mnist"):
            raise ValueError("The MLP model currently supports MNIST and Fashion-MNIST only.")
        return SparseMLP(hidden_dim=args.hidden_dim, sparsity=sparsity)
    if model_name == "cifar_cnn":
        if dataset_name != "cifar10":
            raise ValueError("The CIFAR CNN expects --dataset cifar10.")
        return SageCifarCNN(base_channels=args.base_channels, sparsity=sparsity)
    raise ValueError(f"Unsupported model: {args.model}")


def empty_epoch_growth_stats() -> dict[str, float]:
    return {
        "growth_events": 0.0,
        "pruned_edges": 0.0,
        "grown_edges": 0.0,
        "mean_pruned_weight_magnitude": 0.0,
        "mean_grown_score": 0.0,
    }


def update_epoch_growth_stats(epoch_stats: dict[str, float], growth_stats: dict[str, float]) -> None:
    epoch_stats["growth_events"] += 1.0
    epoch_stats["pruned_edges"] += growth_stats["pruned"]
    epoch_stats["grown_edges"] += growth_stats["grown"]
    epoch_stats["mean_pruned_weight_magnitude"] += (
        growth_stats["mean_pruned_weight_magnitude"] * growth_stats["pruned"]
    )
    epoch_stats["mean_grown_score"] += growth_stats["mean_grown_score"] * growth_stats["grown"]


def finalize_epoch_growth_stats(epoch_stats: dict[str, float]) -> dict[str, float]:
    finalized = dict(epoch_stats)
    if finalized["pruned_edges"] > 0:
        finalized["mean_pruned_weight_magnitude"] /= finalized["pruned_edges"]
    if finalized["grown_edges"] > 0:
        finalized["mean_grown_score"] /= finalized["grown_edges"]
    return finalized


def empty_epoch_neuron_stats() -> dict[str, float]:
    stats = {
        "neuron_prune_events": 0.0,
        "pruned_neurons": 0.0,
        "pruned_hidden1": 0.0,
        "pruned_hidden2": 0.0,
        "mean_pruned_neuron_score": 0.0,
        "pruned_channels": 0.0,
        "mean_pruned_channel_score": 0.0,
    }
    for conv_idx in range(5):
        stats[f"pruned_conv{conv_idx}_channels"] = 0.0
    return stats


def empty_epoch_focus_stats() -> dict[str, float]:
    return {
        "sage_focus_steps": 0.0,
        "mean_boosted_edges": 0.0,
        "mean_weak_scaled_edges": 0.0,
        "mean_boosted_score": 0.0,
    }


def update_epoch_focus_stats(epoch_stats: dict[str, float], focus_stats: dict[str, float]) -> None:
    if focus_stats["boosted_edges"] <= 0:
        return
    epoch_stats["sage_focus_steps"] += 1.0
    epoch_stats["mean_boosted_edges"] += focus_stats["boosted_edges"]
    epoch_stats["mean_weak_scaled_edges"] += focus_stats["weak_scaled_edges"]
    epoch_stats["mean_boosted_score"] += focus_stats["mean_boosted_score"]


def finalize_epoch_focus_stats(epoch_stats: dict[str, float]) -> dict[str, float]:
    finalized = dict(epoch_stats)
    if finalized["sage_focus_steps"] > 0:
        finalized["mean_boosted_edges"] /= finalized["sage_focus_steps"]
        finalized["mean_weak_scaled_edges"] /= finalized["sage_focus_steps"]
        finalized["mean_boosted_score"] /= finalized["sage_focus_steps"]
    return finalized


def should_prune_neurons(args: argparse.Namespace, epoch: int) -> bool:
    prune_end_epoch = args.neuron_prune_end_epoch if args.neuron_prune_end_epoch > 0 else args.epochs
    return (
        args.neuron_prune_fraction > 0.0
        and args.neuron_prune_start_epoch > 0
        and args.neuron_prune_interval > 0
        and epoch >= args.neuron_prune_start_epoch
        and epoch <= prune_end_epoch
        and (epoch - args.neuron_prune_start_epoch) % args.neuron_prune_interval == 0
    )


def should_focus_sage_paths(args: argparse.Namespace, epoch: int) -> bool:
    if args.sage_boost_fraction <= 0.0:
        return False
    if args.sage_grad_boost == 1.0 and args.weak_grad_decay == 1.0:
        return False
    focus_start = args.sage_focus_start_epoch
    if focus_start <= 0:
        focus_start = args.neuron_prune_start_epoch
    focus_end = args.sage_focus_end_epoch
    return epoch >= focus_start and (focus_end <= 0 or epoch <= focus_end)


def should_compact(args: argparse.Namespace, epoch: int) -> bool:
    return args.compact_epoch == epoch or (
        args.compact_epoch == -1 and epoch == args.epochs
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
) -> tuple[int, dict[str, float], dict[str, float]]:
    model.train()
    epoch_growth_stats = empty_epoch_growth_stats()
    epoch_focus_stats = empty_epoch_focus_stats()
    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        if args.train_batches and batch_idx > args.train_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        global_step += 1
        if (
            args.growth_interval > 0
            and args.prune_fraction > 0.0
            and global_step % args.growth_interval == 0
        ):
            growth_stats = model.prune_and_grow(args.prune_fraction, args.growth_mode)
            update_epoch_growth_stats(epoch_growth_stats, growth_stats)

        if should_focus_sage_paths(args, epoch):
            focus_stats = model.apply_sage_gradient_focus(
                boost_factor=args.sage_grad_boost,
                boost_fraction=args.sage_boost_fraction,
                weak_grad_decay=args.weak_grad_decay,
            )
            update_epoch_focus_stats(epoch_focus_stats, focus_stats)

        model.mask_gradients()
        optimizer.step()
        model.apply_mask_to_weights()

    return (
        global_step,
        finalize_epoch_growth_stats(epoch_growth_stats),
        finalize_epoch_focus_stats(epoch_focus_stats),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.numel()

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples


def model_structure_label(model: torch.nn.Module) -> str:
    if hasattr(model, "active_hidden_counts"):
        return f"hidden={model.active_hidden_counts()}"
    if hasattr(model, "active_channel_counts"):
        return f"channels={model.active_channel_counts()}"
    return "structure=unknown"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device()

    train_loader, eval_loader = build_dataloaders(args)
    model_sparsity = 0.0 if args.dense_start else args.sparsity
    model = build_model(args, model_sparsity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = BASE_FIELDNAMES + list(structure_metrics(model).keys()) + list(layerwise_metrics(model).keys())
    global_step = 0

    print(f"Using device: {device}")
    print(f"Model: {resolve_model_name(args)}")
    print(f"Start mode: {'dense' if args.dense_start else 'sparse'}")
    print(f"Initial active parameters: {active_parameter_count(model)}")

    with args.log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        total_epochs = args.epochs + args.post_compact_epochs
        has_compacted = False
        for epoch in range(1, total_epochs + 1):
            global_step, growth_stats, focus_stats = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args,
                global_step,
                epoch,
            )

            neuron_stats = empty_epoch_neuron_stats()
            if should_prune_neurons(args, epoch):
                neuron_stats = model.prune_weak_neurons(
                    prune_fraction=args.neuron_prune_fraction,
                    mode=args.neuron_prune_mode,
                    protect_fraction=args.neuron_protect_fraction,
                    min_hidden_neurons=args.min_hidden_neurons,
                )
                model.apply_mask_to_weights()

            if not has_compacted and should_compact(args, epoch):
                model = model.compact().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                has_compacted = True

            loss, acc = evaluate(model, eval_loader, device, args.eval_batches)
            active_params = active_parameter_count(model)
            concentration = superweight_concentration(model)

            row = {
                "epoch": epoch,
                "loss": f"{loss:.6f}",
                "accuracy": f"{acc:.6f}",
                "active_parameter_count": active_params,
                "superweight_concentration": f"{concentration:.6f}",
                "growth_events": int(growth_stats["growth_events"]),
                "pruned_edges": int(growth_stats["pruned_edges"]),
                "grown_edges": int(growth_stats["grown_edges"]),
                "mean_pruned_weight_magnitude": f"{growth_stats['mean_pruned_weight_magnitude']:.6f}",
                "mean_grown_score": f"{growth_stats['mean_grown_score']:.6f}",
                "neuron_prune_events": int(neuron_stats["neuron_prune_events"]),
                "neuron_prune_mode": args.neuron_prune_mode,
                "pruned_neurons": int(neuron_stats["pruned_neurons"]),
                "pruned_hidden1": int(neuron_stats["pruned_hidden1"]),
                "pruned_hidden2": int(neuron_stats["pruned_hidden2"]),
                "mean_pruned_neuron_score": f"{neuron_stats['mean_pruned_neuron_score']:.6f}",
                "pruned_channels": int(neuron_stats.get("pruned_channels", 0.0)),
                "pruned_conv0_channels": int(neuron_stats.get("pruned_conv0_channels", 0.0)),
                "pruned_conv1_channels": int(neuron_stats.get("pruned_conv1_channels", 0.0)),
                "pruned_conv2_channels": int(neuron_stats.get("pruned_conv2_channels", 0.0)),
                "pruned_conv3_channels": int(neuron_stats.get("pruned_conv3_channels", 0.0)),
                "pruned_conv4_channels": int(neuron_stats.get("pruned_conv4_channels", 0.0)),
                "mean_pruned_channel_score": f"{neuron_stats.get('mean_pruned_channel_score', 0.0):.6f}",
                "compacted": int(has_compacted),
                "sage_focus_steps": int(focus_stats["sage_focus_steps"]),
                "mean_boosted_edges": f"{focus_stats['mean_boosted_edges']:.6f}",
                "mean_weak_scaled_edges": f"{focus_stats['mean_weak_scaled_edges']:.6f}",
                "mean_boosted_score": f"{focus_stats['mean_boosted_score']:.6f}",
            }
            for key, value in structure_metrics(model).items():
                row[key] = int(value)
            for key, value in layerwise_metrics(model).items():
                row[key] = f"{value:.6f}"
            writer.writerow(row)
            log_file.flush()

            print(
                f"epoch={epoch} loss={loss:.4f} acc={acc:.4f} "
                f"active_params={active_params} concentration={concentration:.4f} "
                f"grown={int(growth_stats['grown_edges'])} "
                f"pruned_neurons={int(neuron_stats['pruned_neurons'])} "
                f"focus_steps={int(focus_stats['sage_focus_steps'])} "
                f"{model_structure_label(model)}"
            )

    print(f"Wrote CSV log to {args.log_path}")


if __name__ == "__main__":
    main()
