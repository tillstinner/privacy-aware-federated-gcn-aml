import json
import logging
import time

import torch

from federated_gcn_aml.data.build_graph import build_amlsim_graph, derive_amlsim_targets, load_raw_amlsim
from federated_gcn_aml.experiments.common_amlsim import (
    attach_node_masks,
    create_amlsim_node_masks,
    make_training_metadata,
    parse_args,
    resolve_amlsim_context,
    resolve_device,
    standardize_node_features_train_only,
)
from federated_gcn_aml.models.gcn import GCN
from federated_gcn_aml.training.trainer import train_full_batch


logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    total_run_t0 = time.perf_counter()
    args = parse_args()
    torch.manual_seed(args.seed)

    context = resolve_amlsim_context(args.data_root, args.output_dir)

    raw = load_raw_amlsim(context.data_root)
    sar_set, _ = derive_amlsim_targets(raw.sar_accounts, raw.alert_accounts)
    graph = build_amlsim_graph(
        raw.accounts,
        raw.transactions,
        sar_set,
        representation_version=args.representation_version,
        feature_boundary=args.feature_boundary,
    )
    data = graph.data
    if args.edge_weight_mode == "unit":
        if getattr(data, "edge_weight", None) is None:
            raise ValueError("--edge-weight-mode unit requires a representation with edge weights.")
        data.edge_weight = torch.ones_like(data.edge_weight)
        graph.graph_semantics["edge_weight"] = "unit edge weights ablation"
    logger.info(
        "Constructed AMLSim representation=%s feature_boundary=%s feature_dim_before=%d "
        "feature_dim_after=%d removed_features=%s feature groups=%s final_feature_dim=%d log1p_columns=%s",
        graph.representation_version,
        graph.feature_boundary,
        graph.feature_count_before_boundary,
        graph.feature_count_after_boundary,
        list(graph.actually_removed_feature_columns),
        list(graph.feature_groups.keys()),
        len(graph.feature_columns),
        list(graph.log1p_columns),
    )

    masks = create_amlsim_node_masks(
        data=data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    data = attach_node_masks(data, masks)
    data, scaling = standardize_node_features_train_only(data, graph.feature_columns)
    logger.info(
        "Applied train-only feature standardization to %d features",
        len(graph.feature_columns),
    )

    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        num_layers=args.num_layers,
    )

    device = resolve_device(args.device)
    training_t0 = time.perf_counter()
    trained = train_full_batch(
        model,
        data,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device,
    )
    training_wall_time_sec = time.perf_counter() - training_t0

    torch.save(trained.model.state_dict(), context.output_dir / "model.pt")

    metadata = make_training_metadata(
        args=args,
        context=context,
        data=data,
        device=device,
        graph=graph,
        scaling=scaling,
        metrics=trained.metrics,
        runtime={
            "total_run_wall_time_sec": time.perf_counter() - total_run_t0,
            "training_wall_time_sec": training_wall_time_sec,
            "pretraining_wall_time_sec": None,
        },
    )
    with (context.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved AMLSim GCN outputs to %s", context.output_dir)

if __name__ == "__main__":
    main()
