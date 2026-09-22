import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from federated_gcn_aml.data.build_graph import DEFAULT_AMLSIM_ROOT
from federated_gcn_aml.training.splits import random_node_split_stratified

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AMLSimRunContext:
    data_root: Path
    output_dir: Path


@dataclass(frozen=True)
class AMLSimNodeMasks:
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


@dataclass(frozen=True)
class FeatureScalingArtifacts:
    mean: torch.Tensor
    scale: torch.Tensor
    metadata: dict


def parse_args(description="Train a centralized full-batch GCN on AMLSim outputs."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_AMLSIM_ROOT),
        help="Directory containing AMLSim accounts.csv, transactions.csv, and sar_accounts.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/gcn_amlsim",
        help="Directory for model checkpoint and run metadata.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. 'auto' uses CUDA if available.",
    )
    parser.add_argument(
        "--representation-version",
        default="pass1",
        choices=("pass1", "pass2"),
        help="AMLSim graph/feature representation. pass2 uses aggregated edges and topology-aware features.",
    )
    parser.add_argument(
        "--feature-boundary",
        default="oracle",
        choices=("oracle", "privacy_clean"),
        help="Feature-boundary mode. oracle preserves historical features; privacy_clean drops boundary-violating neighbor features.",
    )
    parser.add_argument(
        "--edge-weight-mode",
        default="original",
        choices=("original", "unit"),
        help="Edge-weight ablation mode. 'unit' replaces existing graph edge weights with ones after graph construction.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return device_arg


def resolve_amlsim_context(arg_data_root, arg_output_dir, create_output_dir=True) -> AMLSimRunContext:
    data_root = Path(arg_data_root).expanduser().resolve()
    output_dir = Path(arg_output_dir).expanduser().resolve()
    if create_output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"data_root: {data_root}")
    logger.info(f"output_dir: {output_dir}")

    return AMLSimRunContext(data_root=data_root, output_dir=output_dir)


def create_amlsim_node_masks(data, train_ratio, val_ratio, test_ratio, seed) -> AMLSimNodeMasks:
    train_mask, val_mask, test_mask = random_node_split_stratified(
        y=data.y,
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
        seed=seed,
    )
    return AMLSimNodeMasks(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def attach_node_masks(data, masks: AMLSimNodeMasks):
    data.train_mask = masks.train_mask
    data.val_mask = masks.val_mask
    data.test_mask = masks.test_mask
    return data


def split_counts(data) -> dict[str, dict[str, int]]:
    """Return split sizes and positive counts for reproducible AMLSim runs."""
    counts = {}
    for split_name in ("train", "val", "test"):
        mask = getattr(data, f"{split_name}_mask")
        counts[split_name] = {
            "count": int(mask.sum().item()),
            "positive_labels": int(data.y[mask].sum().item()),
        }
    return counts


def standardize_node_features_train_only(data, feature_names) -> tuple[object, FeatureScalingArtifacts]:
    """Fit standardization on training nodes only and apply it to all nodes."""
    train_x = data.x[data.train_mask].float()
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    data.x = (data.x.float() - mean) / scale

    metadata = {
        "transform": "standard",
        "fit_on": "train_mask",
        "train_only_fit": True,
        "feature_names": [str(name) for name in feature_names],
        "mean": [float(value) for value in mean.cpu().tolist()],
        "scale": [float(value) for value in scale.cpu().tolist()],
    }
    return data, FeatureScalingArtifacts(mean=mean, scale=scale, metadata=metadata)


def make_training_metadata(args, context: AMLSimRunContext, data, device, graph=None, scaling=None, metrics=None, runtime=None):
    feature_names = list(getattr(graph, "feature_columns", []))
    runtime = dict(runtime or {})
    best_epoch = None if metrics is None else metrics.get("best_epoch")
    return {
        "data_root": str(context.data_root),
        "epochs": args.epochs,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": device,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_node_features": data.num_node_features,
        "representation_version": getattr(graph, "representation_version", None),
        "feature_boundary": getattr(graph, "feature_boundary", None),
        "configured_removed_feature_columns": list(getattr(graph, "configured_removed_feature_columns", ())),
        "actually_removed_feature_columns": list(getattr(graph, "actually_removed_feature_columns", ())),
        "feature_columns_before_boundary": list(getattr(graph, "feature_columns_before_boundary", ())),
        "feature_columns_after_boundary": list(getattr(graph, "feature_columns_after_boundary", feature_names)),
        "feature_count_before_boundary": getattr(graph, "feature_count_before_boundary", None),
        "feature_count_after_boundary": getattr(graph, "feature_count_after_boundary", data.num_node_features),
        "feature_boundary_assumption": getattr(graph, "feature_boundary_assumption", None),
        "graph_semantics": getattr(graph, "graph_semantics", None),
        "edge_weight_mode": getattr(args, "edge_weight_mode", "original"),
        "has_edge_weight": getattr(data, "edge_weight", None) is not None,
        "positive_labels": int((data.y == 1).sum().item()),
        "split_counts": split_counts(data),
        "feature_names": feature_names,
        "feature_groups": getattr(graph, "feature_groups", {}),
        "feature_semantics": getattr(graph, "feature_semantics", {}),
        "log1p_columns": list(getattr(graph, "log1p_columns", ())),
        "scaling": None if scaling is None else scaling.metadata,
        "total_run_wall_time_sec": runtime.get("total_run_wall_time_sec"),
        "training_wall_time_sec": runtime.get("training_wall_time_sec"),
        "pretraining_wall_time_sec": runtime.get("pretraining_wall_time_sec"),
        "best_round_or_epoch": best_epoch,
        "final_round_or_epoch": args.epochs,
        "runtime_measurement_scope": "single_process_runner_wall_time_hardware_dependent",
        "metrics": metrics,
    }
