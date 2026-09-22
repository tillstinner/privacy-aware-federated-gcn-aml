import json
import logging
import time

import torch
from sklearn.linear_model import LogisticRegression

from federated_gcn_aml.data.build_graph import build_amlsim_graph, derive_amlsim_targets, load_raw_amlsim
from federated_gcn_aml.experiments.common_amlsim import (
    attach_node_masks,
    create_amlsim_node_masks,
    make_training_metadata,
    parse_args,
    resolve_amlsim_context,
    standardize_node_features_train_only,
)
from federated_gcn_aml.training.metrics import (
    THRESHOLD_SELECTION_RULE,
    combine_ranking_and_threshold_metrics,
    compute_metrics_from_probabilities,
    select_threshold_from_probabilities,
)


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    total_run_t0 = time.perf_counter()
    args = parse_args(description="Train a logistic-regression baseline on engineered AMLSim node features.")
    torch.manual_seed(args.seed)

    if args.output_dir == "runs/gcn_amlsim":
        args.output_dir = "runs/logreg_amlsim"

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

    x_train = data.x[data.train_mask].cpu().numpy()
    y_train = data.y[data.train_mask].cpu().numpy()
    x_val = data.x[data.val_mask].cpu().numpy()
    y_val = data.y[data.val_mask]
    x_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask]

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=args.seed,
    )
    training_t0 = time.perf_counter()
    model.fit(x_train, y_train)
    training_wall_time_sec = time.perf_counter() - training_t0

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    selected_threshold, val_threshold_metrics = select_threshold_from_probabilities(y_val, val_prob)
    test_threshold_metrics = compute_metrics_from_probabilities(y_test, test_prob, threshold=selected_threshold)
    val_aupr_metrics = compute_metrics_from_probabilities(y_val, val_prob, threshold=0.5)
    test_aupr_metrics = compute_metrics_from_probabilities(y_test, test_prob, threshold=0.5)

    metrics = {
        "selected_threshold": float(selected_threshold),
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "val": combine_ranking_and_threshold_metrics(val_aupr_metrics, val_threshold_metrics),
        "test": combine_ranking_and_threshold_metrics(test_aupr_metrics, test_threshold_metrics),
    }
    logger.info(
        "LogReg baseline finished with val AUPR %.4f, test AUPR %.4f, selected threshold %.2f",
        metrics["val"]["AUPR"],
        metrics["test"]["AUPR"],
        selected_threshold,
    )

    metadata = make_training_metadata(
        args=args,
        context=context,
        data=data,
        device="cpu",
        graph=graph,
        scaling=scaling,
        metrics=metrics,
        runtime={
            "total_run_wall_time_sec": time.perf_counter() - total_run_t0,
            "training_wall_time_sec": training_wall_time_sec,
            "pretraining_wall_time_sec": None,
        },
    )
    metadata["model_type"] = "logistic_regression"
    metadata["model_params"] = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": args.seed,
    }
    with (context.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved AMLSim logistic-regression outputs to %s", context.output_dir)


if __name__ == "__main__":
    main()
