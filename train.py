import argparse
import csv
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.sparse_mlp import SparseMLP
from src.utils.metrics import active_parameter_count, superweight_concentration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SAGE sparse MLP prototype.")
    parser.add_argument("--dataset", choices=["mnist", "fashion-mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--sparsity", type=float, default=0.95)
    parser.add_argument("--growth_mode", choices=["random", "gradient", "sage"], default="sage")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--growth_interval", type=int, default=100)
    parser.add_argument("--prune_fraction", type=float, default=0.05)
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


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    dataset_name = args.dataset.replace("_", "-")
    dataset_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = dataset_cls(args.data_dir, train=True, download=True, transform=transform)
    eval_dataset = dataset_cls(args.data_dir, train=False, download=True, transform=transform)
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


def train_epoch(
    model: SparseMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    global_step: int,
) -> int:
    model.train()
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
            model.prune_and_grow(args.prune_fraction, args.growth_mode)

        model.mask_gradients()
        optimizer.step()
        model.apply_mask_to_weights()

    return global_step


@torch.no_grad()
def evaluate(
    model: SparseMLP,
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device()

    train_loader, eval_loader = build_dataloaders(args)
    model = SparseMLP(hidden_dim=args.hidden_dim, sparsity=args.sparsity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "loss", "accuracy", "active_parameter_count", "superweight_concentration"]
    global_step = 0

    print(f"Using device: {device}")
    print(f"Initial active parameters: {active_parameter_count(model)}")

    with args.log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            global_step = train_epoch(model, train_loader, optimizer, device, args, global_step)
            loss, acc = evaluate(model, eval_loader, device, args.eval_batches)
            active_params = active_parameter_count(model)
            concentration = superweight_concentration(model)

            writer.writerow(
                {
                    "epoch": epoch,
                    "loss": f"{loss:.6f}",
                    "accuracy": f"{acc:.6f}",
                    "active_parameter_count": active_params,
                    "superweight_concentration": f"{concentration:.6f}",
                }
            )
            log_file.flush()

            print(
                f"epoch={epoch} loss={loss:.4f} acc={acc:.4f} "
                f"active_params={active_params} concentration={concentration:.4f}"
            )

    print(f"Wrote CSV log to {args.log_path}")


if __name__ == "__main__":
    main()
