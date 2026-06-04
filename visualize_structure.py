import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SAGE structure/FLOP CSV logs as an SVG.")
    parser.add_argument("log", type=Path, help="CSV log produced by train.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="SVG output path. Defaults to <log stem>_structure.svg.",
    )
    return parser.parse_args()


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def fmt_large(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def svg_text(x: float, y: float, text: str, size: int = 13, weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#172033">{escape(text)}</text>'
    )


def escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def scale(value: float, min_value: float, max_value: float, out_min: float, out_max: float) -> float:
    if max_value <= min_value:
        return (out_min + out_max) / 2
    ratio = (value - min_value) / (max_value - min_value)
    return out_min + ratio * (out_max - out_min)


def chart(
    rows: list[dict[str, str]],
    metrics: list[tuple[str, str, str]],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
) -> list[str]:
    epochs = [as_float(row, "epoch") for row in rows]
    series = [(label, key, color, [as_float(row, key) for row in rows]) for label, key, color in metrics]
    values = [value for _, _, _, metric_values in series for value in metric_values if value > 0]
    max_value = max(values) if values else 1.0
    min_epoch = min(epochs) if epochs else 1.0
    max_epoch = max(epochs) if epochs else 1.0

    parts = [
        svg_text(x, y - 12, title, size=15, weight="700"),
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#cfd6e4" rx="6"/>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#8a94a8"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#8a94a8"/>',
    ]
    for tick in range(1, 4):
        ty = y + height * tick / 4
        parts.append(f'<line x1="{x}" y1="{ty}" x2="{x + width}" y2="{ty}" stroke="#edf0f6"/>')

    legend_x = x + 12
    for idx, (label, _key, color, metric_values) in enumerate(series):
        points = []
        for epoch, value in zip(epochs, metric_values):
            px = scale(epoch, min_epoch, max_epoch, x + 12, x + width - 12)
            py = scale(value, 0.0, max_value, y + height - 12, y + 12)
            points.append(f"{px:.1f},{py:.1f}")
        if len(points) == 1:
            px, py = points[0].split(",")
            parts.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{color}"/>')
        elif points:
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        ly = y + height + 24 + idx * 18
        parts.append(f'<rect x="{legend_x}" y="{ly - 10}" width="10" height="10" fill="{color}"/>')
        parts.append(svg_text(legend_x + 16, ly, f"{label}: {fmt_large(metric_values[-1] if metric_values else 0)}", 12))
    parts.append(svg_text(x + width - 80, y + height + 20, f"max {fmt_large(max_value)}", 12))
    return parts


def draw_matrix_panel(
    row: dict[str, str],
    x: float,
    y: float,
    title: str,
    matrices: list[tuple[str, float, float, float]],
) -> list[str]:
    max_in = max((matrix[1] for matrix in matrices), default=1.0)
    max_out = max((matrix[2] for matrix in matrices), default=1.0)
    parts = [svg_text(x, y, title, size=15, weight="700")]
    top = y + 18
    for idx, (label, input_dim, output_dim, density) in enumerate(matrices):
        rect_x = x + idx * 158
        rect_y = top
        rect_w = max(36.0, 120.0 * input_dim / max_in)
        rect_h = max(28.0, 110.0 * output_dim / max_out)
        shade = int(245 - min(max(density, 0.0), 1.0) * 95)
        fill = f"rgb({shade},{shade + 4},{min(shade + 18, 255)})"
        parts.extend(
            [
                f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{rect_w:.1f}" '
                f'height="{rect_h:.1f}" fill="{fill}" stroke="#44516a" rx="4"/>',
                svg_text(rect_x, rect_y + rect_h + 18, label, 12, "700"),
                svg_text(rect_x, rect_y + rect_h + 34, f"{output_dim:.0f} x {input_dim:.0f}", 12),
                svg_text(rect_x, rect_y + rect_h + 50, f"density {density:.2f}", 12),
            ]
        )
    return parts


def mlp_matrices(row: dict[str, str]) -> list[tuple[str, float, float, float]]:
    hidden1 = as_float(row, "hidden1_dim")
    hidden2 = as_float(row, "hidden2_dim")
    return [
        ("W0", 784.0, hidden1, as_float(row, "layer_0_density", 1.0)),
        ("W1", hidden1, hidden2, as_float(row, "layer_1_density", 1.0)),
        ("W2", hidden2, 10.0, as_float(row, "layer_2_density", 1.0)),
    ]


def cnn_matrices(row: dict[str, str]) -> list[tuple[str, float, float, float]]:
    channels = [as_float(row, f"conv{idx}_channels") for idx in range(5)]
    inputs = [3.0, *channels[:-1]]
    matrices = []
    for idx, (input_channels, output_channels) in enumerate(zip(inputs, channels)):
        matrices.append(
            (
                f"conv{idx}",
                input_channels * 9.0,
                output_channels,
                as_float(row, f"layer_{idx}_density", 1.0),
            )
        )
    matrices.append(
        (
            "head",
            channels[-1],
            10.0,
            as_float(row, "layer_5_density", 1.0),
        )
    )
    return matrices


def structure_metrics(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    final = rows[-1]
    if as_float(final, "total_active_channels") > 0:
        return [
            ("conv0 active", "active_conv0_channels", "#2563eb"),
            ("conv1 active", "active_conv1_channels", "#16a34a"),
            ("conv2 active", "active_conv2_channels", "#dc2626"),
            ("conv3 active", "active_conv3_channels", "#9333ea"),
            ("conv4 active", "active_conv4_channels", "#ea580c"),
        ]
    return [
        ("hidden1 active", "active_hidden1", "#2563eb"),
        ("hidden2 active", "active_hidden2", "#16a34a"),
        ("hidden1 physical", "hidden1_dim", "#93c5fd"),
        ("hidden2 physical", "hidden2_dim", "#86efac"),
    ]


def render_svg(rows: list[dict[str, str]], log_path: Path) -> str:
    first = rows[0]
    final = rows[-1]
    is_cnn = as_float(final, "total_active_channels") > 0
    matrix_builder = cnn_matrices if is_cnn else mlp_matrices
    height = 900
    width = 1180
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f8fc"/>',
        svg_text(32, 42, "SAGE Structure Map", 24, "800"),
        svg_text(32, 68, str(log_path), 13),
        svg_text(
            32,
            94,
            "Physical FLOPs drop only when tensors are compacted; active FLOPs are the ideal masked/sparse work.",
            13,
        ),
    ]
    parts.extend(
        chart(
            rows,
            [
                ("physical forward FLOPs", "physical_forward_flops", "#111827"),
                ("active masked FLOPs", "active_forward_flops", "#2563eb"),
            ],
            32,
            140,
            520,
            230,
            "Estimated Forward FLOPs Per Sample",
        )
    )
    parts.extend(chart(rows, structure_metrics(rows), 620, 140, 520, 230, "Active Structure Over Training"))
    parts.extend(draw_matrix_panel(first, 32, 470, "Initial Physical Matrices", matrix_builder(first)))
    parts.extend(draw_matrix_panel(final, 32, 670, "Final Physical Matrices", matrix_builder(final)))
    parts.extend(
        [
            svg_text(620, 470, "Final Summary", 15, "700"),
            svg_text(620, 502, f"accuracy: {as_float(final, 'accuracy'):.4f}", 13),
            svg_text(620, 526, f"physical params: {fmt_large(as_float(final, 'physical_parameter_count'))}", 13),
            svg_text(620, 550, f"active params: {fmt_large(as_float(final, 'active_parameter_count'))}", 13),
            svg_text(620, 574, f"physical FLOPs: {fmt_large(as_float(final, 'physical_forward_flops'))}", 13),
            svg_text(620, 598, f"active FLOPs: {fmt_large(as_float(final, 'active_forward_flops'))}", 13),
            svg_text(620, 622, f"compacted: {int(as_float(final, 'compacted'))}", 13),
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    output = args.output or args.log.with_name(f"{args.log.stem}_structure.svg")
    with args.log.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{args.log} has no rows.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(rows, args.log))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
