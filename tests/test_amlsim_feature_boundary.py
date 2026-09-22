from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from federated_gcn_aml.data.build_graph import (
    AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS,
    build_amlsim_graph,
)
from federated_gcn_aml.experiments.common_amlsim import (
    AMLSimRunContext,
    attach_node_masks,
    make_training_metadata,
    standardize_node_features_train_only,
)
from federated_gcn_aml.experiments.run_federated_amlsim import _make_metadata as make_federated_metadata
from federated_gcn_aml.federated.graph_builder import build_client_graphs
from federated_gcn_aml.federated.partition import build_bank_partition


def _raw_tables():
    accounts = pd.DataFrame(
        {
            "acct_id": ["a", "b", "c", "d"],
            "bank_id": ["A", "A", "B", "B"],
            "initial_deposit": [100.0, 120.0, 80.0, 90.0],
        }
    )
    transactions = pd.DataFrame(
        {
            "orig_acct": ["a", "a", "b", "c", "d", "d", "c"],
            "bene_acct": ["b", "c", "c", "a", "a", "c", "d"],
            "base_amt": [10.0, 20.0, 5.0, 30.0, 15.0, 25.0, 35.0],
        }
    )
    return accounts, transactions


def _masked_graph(feature_boundary: str):
    accounts, transactions = _raw_tables()
    graph = build_amlsim_graph(
        accounts,
        transactions,
        {"b", "d"},
        representation_version="pass2",
        feature_boundary=feature_boundary,
    )
    num_nodes = int(graph.data.num_nodes)
    graph.data.train_mask = torch.tensor([True, True, False, False])[:num_nodes]
    graph.data.val_mask = torch.tensor([False, False, True, False])[:num_nodes]
    graph.data.test_mask = torch.tensor([False, False, False, True])[:num_nodes]
    return graph


def _central_args(feature_boundary: str):
    return SimpleNamespace(
        epochs=1,
        hidden_channels=4,
        num_layers=2,
        dropout=0.1,
        lr=0.01,
        weight_decay=0.001,
        seed=42,
        edge_weight_mode="original",
        feature_boundary=feature_boundary,
    )


def _federated_args(feature_boundary: str, federated_mode: str = "cut_edges", num_hops: int = 0):
    return SimpleNamespace(
        federated_mode=federated_mode,
        num_hops=num_hops,
        edge_weight_mode="unit",
        feature_pretrain_transport="plain",
        feature_pretrain_payload_protection="none",
        diagnostic_seed=None,
        payload_dp_seed=None,
        model_update_aggregation="plain_fedavg",
        masked_fedavg_seed=None,
        aggregation_weighting="train_nodes",
        rounds=1,
        local_epochs=1,
        hidden_channels=4,
        num_layers=2,
        dropout=0.1,
        lr=0.01,
        weight_decay=0.001,
        seed=42,
        device="cpu",
        feature_pretrain_block_rows=64,
        feature_boundary=feature_boundary,
    )


def _result(model_update_aggregation="plain_fedavg"):
    diagnostics = {"pretrain_protocol": {}, "model_update_aggregation": {}}
    if model_update_aggregation == "masked_fedavg":
        diagnostics["model_update_aggregation"] = {
            "secure_aggregation": True,
            "plaintext_model_updates_visible_to_server": False,
            "masked_model_updates_visible_to_server": True,
            "aggregate_update_visible_to_server": True,
            "masked_fedavg_vector_numel": 6,
        }
    return SimpleNamespace(
        diagnostics=diagnostics,
        metrics={"actual_val_count": 1, "actual_test_count": 1, "best_round": 1},
        model_state={
            "w": torch.zeros((2, 3), dtype=torch.float32),
            "counter": torch.tensor(1, dtype=torch.long),
        },
    )


def _partition_and_client_graphs(graph, mode: str, num_hops: int):
    accounts, _ = _raw_tables()
    split_masks = {
        "train": graph.data.train_mask,
        "val": graph.data.val_mask,
        "test": graph.data.test_mask,
    }
    partition = build_bank_partition(accounts, graph.account_ids, graph.id2idx, y=graph.data.y, masks=split_masks)
    client_graphs = build_client_graphs(
        global_data=graph.data,
        partition=partition,
        mode=mode,
        num_hops=num_hops,
        edge_weight_mode="unit",
    )
    return partition, client_graphs


def test_oracle_keeps_old_pass2_feature_count():
    graph = _masked_graph("oracle")
    assert graph.feature_boundary == "oracle"
    assert graph.feature_count_before_boundary == 36
    assert graph.feature_count_after_boundary == 36
    assert graph.configured_removed_feature_columns == ()
    assert graph.actually_removed_feature_columns == ()
    assert graph.data.x.size(1) == len(graph.feature_columns)


def test_privacy_clean_drops_exactly_configured_pass2_features():
    graph = _masked_graph("privacy_clean")
    removed = set(AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS)
    assert graph.feature_boundary == "privacy_clean"
    assert set(graph.configured_removed_feature_columns) == removed
    assert set(graph.actually_removed_feature_columns) == removed
    assert graph.feature_count_before_boundary == 36
    assert graph.feature_count_after_boundary == 32
    assert len(graph.feature_columns_before_boundary) - len(graph.feature_columns_after_boundary) == 4
    assert removed.isdisjoint(graph.feature_columns_after_boundary)
    assert set(graph.feature_columns_before_boundary) - set(graph.feature_columns_after_boundary) == removed
    assert "neighbor_activity" not in graph.feature_groups
    assert removed.isdisjoint(graph.feature_semantics)
    assert removed.isdisjoint(graph.log1p_columns)
    assert graph.data.x.size(1) == len(graph.feature_columns) == graph.feature_count_after_boundary


def test_privacy_clean_does_not_fail_when_configured_columns_are_absent():
    accounts, transactions = _raw_tables()
    graph = build_amlsim_graph(
        accounts,
        transactions,
        {"b", "d"},
        representation_version="pass1",
        feature_boundary="privacy_clean",
    )
    assert set(graph.configured_removed_feature_columns) == set(AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS)
    assert graph.actually_removed_feature_columns == ()
    assert graph.feature_count_before_boundary == graph.feature_count_after_boundary == len(graph.feature_columns)
    assert graph.data.x.size(1) == len(graph.feature_columns)


def test_scaling_metadata_matches_active_feature_columns():
    graph = _masked_graph("privacy_clean")
    data, scaling = standardize_node_features_train_only(graph.data, graph.feature_columns)
    assert data.x.size(1) == len(graph.feature_columns)
    assert scaling.metadata["feature_names"] == list(graph.feature_columns)
    assert len(scaling.metadata["mean"]) == len(graph.feature_columns)
    assert len(scaling.metadata["scale"]) == len(graph.feature_columns)


def test_centralized_metadata_records_feature_boundary_fields():
    graph = _masked_graph("privacy_clean")
    data, scaling = standardize_node_features_train_only(graph.data, graph.feature_columns)
    metadata = make_training_metadata(
        args=_central_args("privacy_clean"),
        context=AMLSimRunContext(data_root=Path("/tmp/data"), output_dir=Path("/tmp/out")),
        data=data,
        device="cpu",
        graph=graph,
        scaling=scaling,
        metrics={},
        runtime={"total_run_wall_time_sec": 3.0, "training_wall_time_sec": 2.0, "pretraining_wall_time_sec": None},
    )
    assert metadata["feature_boundary"] == "privacy_clean"
    assert metadata["configured_removed_feature_columns"] == list(AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS)
    assert metadata["actually_removed_feature_columns"] == list(AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS)
    assert metadata["feature_count_before_boundary"] == 36
    assert metadata["feature_count_after_boundary"] == 32
    assert metadata["feature_names"] == metadata["feature_columns_after_boundary"]
    assert metadata["total_run_wall_time_sec"] == 3.0
    assert metadata["training_wall_time_sec"] == 2.0
    assert metadata["pretraining_wall_time_sec"] is None
    assert metadata["best_round_or_epoch"] is None
    assert metadata["final_round_or_epoch"] == 1
    assert metadata["runtime_measurement_scope"] == "single_process_runner_wall_time_hardware_dependent"


def test_federated_metadata_records_feature_boundary_fields():
    graph = _masked_graph("privacy_clean")
    data, scaling = standardize_node_features_train_only(graph.data, graph.feature_columns)
    partition, client_graphs = _partition_and_client_graphs(graph, "cut_edges", 0)
    metadata = make_federated_metadata(
        args=_federated_args("privacy_clean"),
        context=AMLSimRunContext(data_root=Path("/tmp/data"), output_dir=Path("/tmp/out")),
        graph=graph,
        data=data,
        scaling=scaling,
        partition=partition,
        client_graphs=client_graphs,
        result=_result(),
        model_class_name="GCN",
        runtime={"total_run_wall_time_sec": 3.0, "training_wall_time_sec": 2.0, "pretraining_wall_time_sec": None},
    )
    assert metadata["feature_boundary"] == "privacy_clean"
    assert metadata["configured_removed_feature_columns"] == list(AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS)
    assert metadata["actually_removed_feature_columns"] == list(AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS)
    assert metadata["feature_count_before_boundary"] == 36
    assert metadata["feature_count_after_boundary"] == 32
    assert metadata["feature_names"] == metadata["feature_columns_after_boundary"]
    assert metadata["total_run_wall_time_sec"] == 3.0
    assert metadata["training_wall_time_sec"] == 2.0
    assert metadata["pretraining_wall_time_sec"] is None
    assert metadata["best_round_or_epoch"] == 1
    assert metadata["final_round_or_epoch"] == 1
    assert metadata["runtime_measurement_scope"] == "single_process_runner_wall_time_hardware_dependent"
    assert metadata["model_update_num_parameters"] == 6
    assert metadata["model_update_state_numel"] == 7
    assert metadata["model_update_state_bytes_plain_estimate"] == 32
    assert metadata["model_update_non_floating_state_bytes_estimate"] == 8
    assert metadata["model_update_bytes_per_client_per_round_plain_estimate"] == 32
    assert metadata["model_update_total_upload_bytes_plain_estimate"] == 32
    assert metadata["model_update_global_broadcast_bytes_per_client_per_round_estimate"] == 32
    assert metadata["model_update_total_global_broadcast_download_bytes_estimate"] == 64
    assert metadata["model_update_total_rounds"] == 1
    assert metadata["masked_fedavg_vector_size"] is None
    assert metadata["model_update_bytes_per_client_per_round_masked_estimate"] is None
    assert metadata["model_update_total_upload_bytes_masked_estimate"] is None
    assert metadata["topology_privacy"] is False
    assert metadata["server_visible_plaintext_feature_contributions"] is None
    assert metadata["plaintext_model_updates_visible_to_server"] is True
    assert metadata["masked_model_updates_visible_to_server"] is False
    assert metadata["aggregate_update_visible_to_server"] is True


def test_federated_metadata_records_masked_update_and_pretrain_visibility_fields():
    graph = _masked_graph("privacy_clean")
    data, scaling = standardize_node_features_train_only(graph.data, graph.feature_columns)
    partition, client_graphs = _partition_and_client_graphs(graph, "fedgcn_feature_pretrain", 1)
    args = _federated_args("privacy_clean", "fedgcn_feature_pretrain", 1)
    args.model_update_aggregation = "masked_fedavg"
    args.feature_pretrain_transport = "he"
    metadata = make_federated_metadata(
        args=args,
        context=AMLSimRunContext(data_root=Path("/tmp/data"), output_dir=Path("/tmp/out")),
        graph=graph,
        data=data,
        scaling=scaling,
        partition=partition,
        client_graphs=client_graphs,
        result=_result("masked_fedavg"),
        model_class_name="PreAggregatedGCN",
        runtime={"total_run_wall_time_sec": 3.0, "training_wall_time_sec": 2.0, "pretraining_wall_time_sec": 1.0},
    )
    assert metadata["server_visible_plaintext_feature_contributions"] is False
    assert metadata["plaintext_model_updates_visible_to_server"] is False
    assert metadata["masked_model_updates_visible_to_server"] is True
    assert metadata["aggregate_update_visible_to_server"] is True
    assert metadata["masked_fedavg_vector_size"] == 6
    assert metadata["model_update_bytes_per_client_per_round_masked_estimate"] == 56
    assert metadata["model_update_total_upload_bytes_masked_estimate"] == 56


def test_privacy_clean_client_graph_construction_works_for_cut_edges_h1_h2():
    for mode, hops in (("cut_edges", 0), ("fedgcn_feature_pretrain", 1), ("fedgcn_feature_pretrain", 2)):
        graph = _masked_graph("privacy_clean")
        partition, client_graphs = _partition_and_client_graphs(graph, mode, hops)
        assert partition.diagnostics["num_clients"] == 2
        assert client_graphs
        for client_graph in client_graphs:
            assert client_graph.data.x.size(1) == graph.feature_count_after_boundary
