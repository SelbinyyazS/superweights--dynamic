import argparse
import csv
import re
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SAGE experiment CSV logs.")
    parser.add_argument("logs", nargs="*", type=Path, help="CSV log files. Defaults to logs/*.csv.")
    parser.add_argument("--aggregate", action="store_true", help="Group runs by mode and report mean/std.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def infer_mode(path: Path, final_row: dict[str, str]) -> str:
    name = path.stem.lower()
    if "dense" in name:
        return "dense"
    if "magnitude" in name:
        return "magnitude"
    if "random" in name:
        return "random"
    if "sage" in name:
        return "sage"

    logged_mode = final_row.get("neuron_prune_mode", "")
    if logged_mode:
        return logged_mode
    return "unknown"


def infer_seed(path: Path) -> str:
    match = re.search(r"seed[_-]?(\d+)", path.stem.lower())
    if match is None:
        return ""
    return match.group(1)


def summarize_log(path: Path) -> dict[str, float | str]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} has no data rows.")

    first = rows[0]
    final = rows[-1]
    best = max(rows, key=lambda row: as_float(row, "accuracy"))

    initial_physical = as_float(first, "physical_parameter_count")
    final_physical = as_float(final, "physical_parameter_count")
    reduction = 0.0
    if initial_physical > 0:
        reduction = 1.0 - final_physical / initial_physical

    return {
        "log": str(path),
        "mode": infer_mode(path, final),
        "seed": infer_seed(path),
        "final_epoch": as_float(final, "epoch"),
        "final_accuracy": as_float(final, "accuracy"),
        "best_accuracy": as_float(best, "accuracy"),
        "best_epoch": as_float(best, "epoch"),
        "final_loss": as_float(final, "loss"),
        "final_active_params": as_float(final, "active_parameter_count"),
        "final_physical_params": final_physical,
        "physical_reduction": reduction,
        "hidden1_dim": as_float(final, "hidden1_dim"),
        "hidden2_dim": as_float(final, "hidden2_dim"),
        "active_hidden1": as_float(final, "active_hidden1"),
        "active_hidden2": as_float(final, "active_hidden2"),
        "total_active_channels": as_float(final, "total_active_channels"),
        "conv0_channels": as_float(final, "conv0_channels"),
        "conv1_channels": as_float(final, "conv1_channels"),
        "conv2_channels": as_float(final, "conv2_channels"),
        "conv3_channels": as_float(final, "conv3_channels"),
        "conv4_channels": as_float(final, "conv4_channels"),
        "concentration": as_float(final, "superweight_concentration"),
        "compacted": as_float(final, "compacted"),
    }


def format_value(header: str, value: float | str) -> str:
    if isinstance(value, float):
        if header.endswith("std"):
            return f"{value:.4f}"
        if "reduction" in header:
            return f"{value:.2%}"
        if (
            header.endswith("accuracy")
            or "accuracy_" in header
            or "concentration" in header
        ):
            return f"{value:.4f}"
        return f"{value:.0f}"
    return value


def render_rows(headers: list[str], rows: list[dict[str, float | str]]) -> str:
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(format_value(header, row[header]) for header in headers))
    return "\n".join(lines)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def aggregate_summaries(summaries: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    by_mode: dict[str, list[dict[str, float | str]]] = {}
    for summary in summaries:
        by_mode.setdefault(str(summary["mode"]), []).append(summary)

    rows = []
    for mode, mode_summaries in sorted(by_mode.items()):
        final_acc = [float(summary["final_accuracy"]) for summary in mode_summaries]
        best_acc = [float(summary["best_accuracy"]) for summary in mode_summaries]
        final_params = [float(summary["final_physical_params"]) for summary in mode_summaries]
        reductions = [float(summary["physical_reduction"]) for summary in mode_summaries]
        concentrations = [float(summary["concentration"]) for summary in mode_summaries]
        hidden1 = [float(summary["hidden1_dim"]) for summary in mode_summaries]
        hidden2 = [float(summary["hidden2_dim"]) for summary in mode_summaries]
        active_channels = [float(summary["total_active_channels"]) for summary in mode_summaries]
        rows.append(
            {
                "mode": mode,
                "n": float(len(mode_summaries)),
                "final_accuracy_mean": mean(final_acc),
                "final_accuracy_std": std(final_acc),
                "best_accuracy_mean": mean(best_acc),
                "best_accuracy_std": std(best_acc),
                "final_physical_params_mean": mean(final_params),
                "physical_reduction_mean": mean(reductions),
                "hidden1_dim_mean": mean(hidden1),
                "hidden2_dim_mean": mean(hidden2),
                "total_active_channels_mean": mean(active_channels),
                "concentration_mean": mean(concentrations),
                "concentration_std": std(concentrations),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    paths = args.logs or sorted(Path("logs").glob("*.csv"))
    summaries = [summarize_log(path) for path in paths]

    if args.aggregate:
        headers = [
            "mode",
            "n",
            "final_accuracy_mean",
            "final_accuracy_std",
            "best_accuracy_mean",
            "best_accuracy_std",
            "final_physical_params_mean",
            "physical_reduction_mean",
            "hidden1_dim_mean",
            "hidden2_dim_mean",
            "total_active_channels_mean",
            "concentration_mean",
            "concentration_std",
        ]
        output = render_rows(headers, aggregate_summaries(summaries))
    else:
        headers = [
            "log",
            "mode",
            "seed",
            "final_accuracy",
            "best_accuracy",
            "best_epoch",
            "final_physical_params",
            "physical_reduction",
            "hidden1_dim",
            "hidden2_dim",
            "total_active_channels",
            "conv0_channels",
            "conv1_channels",
            "conv2_channels",
            "conv3_channels",
            "conv4_channels",
            "concentration",
            "compacted",
        ]
        output = render_rows(headers, summaries)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
