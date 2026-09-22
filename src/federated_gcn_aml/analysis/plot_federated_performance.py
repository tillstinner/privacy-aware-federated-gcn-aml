from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = {
    "AUPR": "test_AUPR",
    "F1": "test_F1",
    "MCC": "test_MCC",
}


@dataclass(frozen=True)
class RunSeries:
    label: str
    path: Path
    rounds: list[float]
    local_steps: list[float]
    metrics: dict[str, list[float]]


@dataclass(frozen=True)
class ReferenceLine:
    label: str
    path: Path
    metrics: dict[str, float]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot federated AMLSim validation/test performance trajectories from round_metrics.csv files."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="Directory to scan recursively for round_metrics.csv. May be passed multiple times.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Exact federated run directory containing round_metrics.csv. May be passed multiple times.",
    )
    parser.add_argument(
        "--centralized-run",
        action="append",
        default=[],
        help="Centralized metadata.json file to draw as a horizontal reference line. May be passed multiple times.",
    )
    parser.add_argument(
        "--centralized-label",
        action="append",
        default=[],
        help="Optional label for each --centralized-run, in the same order.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for plots and manifest.")
    parser.add_argument(
        "--x-axis",
        default="round",
        choices=("round", "local_steps"),
        help="Use communication round or approximate local optimizer epochs on the x axis.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "val"),
        help="Metric split to plot. Default is test for final experiment comparison.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Federated AMLSim pass2",
        help="Prefix used in plot titles.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only include runs whose relative path or generated label contains this substring. May be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude runs whose relative path or generated label contains this substring. May be repeated.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def _label_for_run(run_dir: Path, scan_roots: list[Path], metadata: dict) -> str:
    try:
        nearest_root = max((root for root in scan_roots if run_dir.is_relative_to(root)), key=lambda root: len(root.parts))
        relative = run_dir.relative_to(nearest_root)
        root_name = nearest_root.name
    except ValueError:
        relative = run_dir
        root_name = run_dir.parent.name

    mode = metadata.get("federated_mode")
    hops = metadata.get("num_hops")
    rounds = metadata.get("rounds")
    local_epochs = metadata.get("local_epochs")
    lr = metadata.get("lr")

    relative_parts = relative.parts if isinstance(relative, Path) else ()
    sweep_name = relative_parts[0] if relative_parts else run_dir.parents[1].name

    if mode is None:
        return f"{root_name}/{relative}"

    parts = [str(sweep_name), str(mode), f"h{hops}", f"r{rounds}", f"le{local_epochs}"]
    if lr is not None:
        parts.append(f"lr{lr:g}")
    return " ".join(parts)


def _passes_filters(label: str, relative_path: str, include: list[str], exclude: list[str]) -> bool:
    haystack = f"{label} {relative_path}"
    if include and not all(token in haystack for token in include):
        return False
    return not any(token in haystack for token in exclude)


def _read_run_series(run_dir: Path, scan_roots: list[Path], split: str) -> RunSeries:
    metric_columns = {name: f"{split}_{name}" for name in METRICS}
    metrics_path = run_dir / "round_metrics.csv"
    metadata = _read_json(run_dir / "metadata.json")
    label = _label_for_run(run_dir, scan_roots, metadata)

    local_epochs = float(metadata.get("local_epochs", 1))
    rounds: list[float] = []
    local_steps: list[float] = []
    metric_values = {name: [] for name in METRICS}

    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            round_idx = float(row["round"])
            rounds.append(round_idx)
            local_steps.append(round_idx * local_epochs)
            for name, column in metric_columns.items():
                metric_values[name].append(float(row[column]))

    return RunSeries(
        label=label,
        path=run_dir,
        rounds=rounds,
        local_steps=local_steps,
        metrics=metric_values,
    )


def discover_runs(
    scan_roots: list[Path],
    exact_run_dirs: list[Path],
    split: str,
    include: list[str],
    exclude: list[str],
) -> list[RunSeries]:
    runs: list[RunSeries] = []

    for run_dir in exact_run_dirs:
        run = _read_run_series(run_dir, scan_roots + exact_run_dirs, split)
        if _passes_filters(run.label, str(run_dir), include, exclude):
            runs.append(run)

    for root in scan_roots:
        for metrics_path in sorted(root.rglob("round_metrics.csv")):
            run_dir = metrics_path.parent
            run = _read_run_series(run_dir, scan_roots, split)
            label = run.label
            relative_path = str(run_dir.relative_to(root))
            if not _passes_filters(label, relative_path, include, exclude):
                continue
            runs.append(run)

    return sorted(runs, key=lambda run: run.label)


def read_references(paths: list[Path], labels: list[str], split: str) -> list[ReferenceLine]:
    references = []
    for idx, path in enumerate(paths):
        metadata = _read_json(path)
        label = labels[idx] if idx < len(labels) else path.parent.parent.name
        metrics = {
            name: float(metadata["metrics"][split][name])
            for name in METRICS
        }
        references.append(ReferenceLine(label=label, path=path, metrics=metrics))
    return references


def write_manifest(path: Path, runs: list[RunSeries], references: list[ReferenceLine]) -> None:
    rows = {
        "runs": [
            {
                "label": run.label,
                "path": str(run.path),
                "num_points": len(run.rounds),
                "final_round": run.rounds[-1] if run.rounds else None,
            }
            for run in runs
        ],
        "references": [
            {
                "label": reference.label,
                "path": str(reference.path),
                "metrics": reference.metrics,
            }
            for reference in references
        ],
    }
    with path.open("w") as handle:
        json.dump(rows, handle, indent=2)


def plot_metric(
    output_path: Path,
    runs: list[RunSeries],
    references: list[ReferenceLine],
    metric_name: str,
    x_axis: str,
    title_prefix: str,
    split: str,
) -> None:
    plt.figure(figsize=(14, 8))
    for run in runs:
        x_values = run.local_steps if x_axis == "local_steps" else run.rounds
        plt.plot(x_values, run.metrics[metric_name], linewidth=1.8, alpha=0.9, label=run.label)
    for reference in references:
        plt.axhline(
            reference.metrics[metric_name],
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
            label=reference.label,
        )

    xlabel = "Local optimizer epochs (round x local_epochs)" if x_axis == "local_steps" else "Federated round"
    plt.xlabel(xlabel)
    plt.ylabel(f"{split.upper()} {metric_name}")
    plt.title(f"{title_prefix}: {split.upper()} {metric_name} over training")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    scan_roots = [Path(root).expanduser().resolve() for root in args.run_root]
    exact_run_dirs = [Path(run_dir).expanduser().resolve() for run_dir in args.run_dir]
    centralized_paths = [Path(path).expanduser().resolve() for path in args.centralized_run]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = {
        "AUPR": f"{args.split}_AUPR",
        "F1": f"{args.split}_F1",
        "MCC": f"{args.split}_MCC",
    }
    METRICS.clear()
    METRICS.update(metric_keys)

    if not scan_roots and not exact_run_dirs:
        raise SystemExit("Pass at least one --run-root or --run-dir.")

    runs = discover_runs(scan_roots, exact_run_dirs, args.split, args.include, args.exclude)
    if not runs:
        raise SystemExit("No round_metrics.csv files matched the requested roots/filters.")
    references = read_references(centralized_paths, args.centralized_label, args.split)

    write_manifest(output_dir / "federated_performance_manifest.json", runs, references)
    for metric_name in METRICS:
        plot_metric(
            output_path=output_dir / f"{args.split}_{metric_name.lower()}_{args.x_axis}.png",
            runs=runs,
            references=references,
            metric_name=metric_name,
            x_axis=args.x_axis,
            title_prefix=args.title_prefix,
            split=args.split,
        )

    print(f"Wrote {len(METRICS)} plots for {len(runs)} runs and {len(references)} references to {output_dir}")


if __name__ == "__main__":
    main()
