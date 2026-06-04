import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense/SAGE/magnitude/random experiments across seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["dense", "sage", "magnitude", "random"],
        default=["dense", "sage", "magnitude", "random"],
    )
    parser.add_argument("--dataset", choices=["mnist", "fashion-mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("logs/multiseed"))
    parser.add_argument("--neuron_prune_start_epoch", type=int, default=5)
    parser.add_argument("--neuron_prune_interval", type=int, default=1)
    parser.add_argument("--neuron_prune_fraction", type=float, default=0.05)
    parser.add_argument("--neuron_protect_fraction", type=float, default=0.05)
    parser.add_argument("--min_hidden_neurons", type=int, default=8)
    parser.add_argument("--compact_epoch", type=int, default=-1)
    parser.add_argument("--train_batches", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python executable used to launch train.py. Defaults to .venv/bin/python if present.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_python(args: argparse.Namespace) -> str:
    if args.python is not None:
        return str(args.python)
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def build_command(args: argparse.Namespace, mode: str, seed: int, log_path: Path) -> list[str]:
    command = [
        resolve_python(args),
        "train.py",
        "--dataset",
        args.dataset,
        "--dense_start",
        "--growth_interval",
        "0",
        "--epochs",
        str(args.epochs),
        "--hidden_dim",
        str(args.hidden_dim),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--data_dir",
        str(args.data_dir),
        "--log_path",
        str(log_path),
        "--seed",
        str(seed),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.train_batches > 0:
        command.extend(["--train_batches", str(args.train_batches)])
    if args.eval_batches > 0:
        command.extend(["--eval_batches", str(args.eval_batches)])

    if mode != "dense":
        command.extend(
            [
                "--neuron_prune_start_epoch",
                str(args.neuron_prune_start_epoch),
                "--neuron_prune_interval",
                str(args.neuron_prune_interval),
                "--neuron_prune_fraction",
                str(args.neuron_prune_fraction),
                "--neuron_prune_mode",
                mode,
                "--neuron_protect_fraction",
                str(args.neuron_protect_fraction),
                "--min_hidden_neurons",
                str(args.min_hidden_neurons),
                "--compact_epoch",
                str(args.compact_epoch),
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for mode in args.modes:
            log_path = args.output_dir / f"{mode}_seed{seed}.csv"
            if args.skip_existing and log_path.exists():
                print(f"skip existing {log_path}")
                continue

            command = build_command(args, mode, seed, log_path)
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
