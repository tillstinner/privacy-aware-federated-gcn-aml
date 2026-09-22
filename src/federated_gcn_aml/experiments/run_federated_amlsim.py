from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch

from federated_gcn_aml.data.build_graph import DEFAULT_AMLSIM_ROOT, build_amlsim_graph, derive_amlsim_targets, load_raw_amlsim
from federated_gcn_aml.experiments.common_amlsim import (
    attach_node_masks,
    create_amlsim_node_masks,
    resolve_amlsim_context,
    resolve_device,
    split_counts,
    standardize_node_features_train_only,
)
from federated_gcn_aml.federated.client import FederatedClient
from federated_gcn_aml.federated.evaluator import GlobalOwnedNodeEvaluator
from federated_gcn_aml.federated.graph_builder import build_client_graphs
from federated_gcn_aml.federated.partition import build_bank_partition
from federated_gcn_aml.federated.payload_protection import (
    DIAGNOSTIC_PERTURBATION_MODES,
    FORMAL_ROW_DP_MODES,
    SUPPORTED_PAYLOAD_PROTECTION_MODES,
    UNSUPPORTED_PAYLOAD_PROTECTION_MODES,
)
from federated_gcn_aml.federated.protocols import FedGCNFeaturePretrainProtocol, NoCommunicationProtocol
from federated_gcn_aml.federated.server import FederatedServer
from federated_gcn_aml.federated.strategies import FedAvgStrategy
from federated_gcn_aml.models.gcn import GCN, PreAggregatedGCN


logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run protocol-shaped federated GCN training on AMLSim pass2 data.")
    parser.add_argument("--data-root", default=str(DEFAULT_AMLSIM_ROOT))
    parser.add_argument("--output-dir", default="runs/amlsim_federated")
    parser.add_argument("--representation-version", default="pass2", choices=("pass2",))
    parser.add_argument(
        "--feature-boundary",
        default="oracle",
        choices=("oracle", "privacy_clean"),
        help="Feature-boundary mode. oracle preserves historical features; privacy_clean drops boundary-violating neighbor features.",
    )
    parser.add_argument("--federated-mode", default="cut_edges", choices=("cut_edges", "fedgcn_feature_pretrain"))
    parser.add_argument("--num-hops", type=int, default=0)
    parser.add_argument("--edge-weight-mode", default="unit", choices=("unit", "original"))
    parser.add_argument("--feature-pretrain-transport", default="plain", choices=("plain", "he"))
    parser.add_argument(
        "--feature-pretrain-payload-protection",
        default="none",
        choices=tuple(sorted(SUPPORTED_PAYLOAD_PROTECTION_MODES)),
        help=(
            "Payload release mechanism. Diagnostic perturbation modes do not make a formal DP claim; "
            "*_row_dp modes use calibrated contribution-row DP."
        ),
    )
    parser.add_argument("--feature-pretrain-block-rows", type=int, default=64)
    parser.add_argument("--payload-clip-norm", type=float, default=None, help="L2 row-clipping norm.")
    parser.add_argument(
        "--diagnostic-noise-multiplier",
        type=float,
        default=None,
        help="Gaussian diagnostic noise standard deviation divided by the clipping norm.",
    )
    parser.add_argument("--diagnostic-seed", type=int, default=None)
    parser.add_argument("--payload-dp-epsilon", type=float, default=None, help="Formal row-DP epsilon.")
    parser.add_argument("--payload-dp-delta", type=float, default=None, help="Formal row-DP delta.")
    parser.add_argument(
        "--payload-dp-calibration",
        default="classical_gaussian",
        choices=("classical_gaussian", "analytic_gaussian"),
    )
    parser.add_argument("--payload-dp-seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--aggregation-weighting", default="train_nodes", choices=("train_nodes", "equal"))
    parser.add_argument("--model-update-aggregation", default="plain_fedavg", choices=("plain_fedavg", "masked_fedavg"))
    parser.add_argument("--masked-fedavg-seed", type=int, default=None)
    parser.add_argument("--masked-fedavg-min-clients", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args) -> None:
    args.feature_pretrain_payload_protection_requested = args.feature_pretrain_payload_protection
    if args.federated_mode == "cut_edges":
        if args.num_hops not in {0, 1}:
            raise ValueError("cut_edges accepts --num-hops 0; legacy hop values are ignored only for 0 or 1.")
        args.num_hops = 0
        if args.feature_pretrain_transport != "plain" or args.feature_pretrain_payload_protection != "none":
            raise ValueError("cut_edges does not run feature-pretraining transport or payload-protection settings.")
    elif args.federated_mode == "fedgcn_feature_pretrain":
        if args.num_hops not in {1, 2}:
            raise ValueError("fedgcn_feature_pretrain requires --num-hops 1 or --num-hops 2.")
        if args.feature_pretrain_payload_protection in UNSUPPORTED_PAYLOAD_PROTECTION_MODES:
            raise NotImplementedError(
                f"feature-pretrain payload protection {args.feature_pretrain_payload_protection!r} is planned but not implemented. "
                "Supported policies are 'none', 'central_diagnostic_perturbation', "
                "'local_diagnostic_perturbation', 'central_row_dp', and 'local_row_dp'."
            )
        if args.feature_pretrain_payload_protection in DIAGNOSTIC_PERTURBATION_MODES:
            if args.payload_clip_norm is None or args.payload_clip_norm <= 0:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} requires --payload-clip-norm > 0."
                )
            if args.diagnostic_noise_multiplier is None or args.diagnostic_noise_multiplier < 0:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} requires --diagnostic-noise-multiplier >= 0."
                )
            if args.diagnostic_seed is None:
                args.diagnostic_seed = int(args.seed + 1_000_003)
            if args.payload_dp_epsilon is not None or args.payload_dp_delta is not None:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} is diagnostic and does not accept "
                    "--payload-dp-epsilon or --payload-dp-delta."
                )
        if args.feature_pretrain_payload_protection in FORMAL_ROW_DP_MODES:
            if args.payload_clip_norm is None or args.payload_clip_norm <= 0:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} requires --payload-clip-norm > 0."
                )
            if args.payload_dp_epsilon is None or args.payload_dp_epsilon <= 0:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} requires --payload-dp-epsilon > 0."
                )
            if args.payload_dp_calibration == "classical_gaussian" and args.payload_dp_epsilon > 1.0:
                raise ValueError(
                    "classical Gaussian contribution-row DP calibration requires --payload-dp-epsilon <= 1.0."
                )
            if args.payload_dp_delta is None or args.payload_dp_delta <= 0 or args.payload_dp_delta >= 1:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} requires --payload-dp-delta in (0, 1)."
                )
            if args.diagnostic_noise_multiplier is not None:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} uses calibrated sigma and does not accept "
                    "--diagnostic-noise-multiplier."
                )
            if args.diagnostic_seed is not None:
                raise ValueError(
                    f"{args.feature_pretrain_payload_protection} uses --payload-dp-seed and does not accept "
                    "--diagnostic-seed."
                )
            if args.payload_dp_seed is None:
                args.payload_dp_seed = int(args.seed + 1_000_003)
    else:
        raise ValueError(f"Unsupported federated mode: {args.federated_mode}")
    if args.feature_pretrain_block_rows <= 0:
        raise ValueError("--feature-pretrain-block-rows must be positive.")
    if args.masked_fedavg_min_clients <= 0:
        raise ValueError("--masked-fedavg-min-clients must be positive.")
    if args.model_update_aggregation == "masked_fedavg":
        if args.masked_fedavg_min_clients < 2:
            raise ValueError("masked_fedavg requires --masked-fedavg-min-clients >= 2.")
        if args.masked_fedavg_seed is None:
            args.masked_fedavg_seed = int(args.seed + 2_000_003)


def select_model_class(federated_mode: str, num_hops: int):
    if federated_mode == "cut_edges":
        return GCN
    if federated_mode == "fedgcn_feature_pretrain" and num_hops == 1:
        return PreAggregatedGCN
    if federated_mode == "fedgcn_feature_pretrain" and num_hops == 2:
        return PreAggregatedGCN
    raise ValueError(f"Unsupported model mapping: mode={federated_mode} num_hops={num_hops}")


def _split_masks_dict(data) -> dict[str, torch.Tensor]:
    return {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}


def _write_round_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def external_reference_mapping() -> list[dict]:
    return [
        {
            "concept": "FedGCN communication index / client subgraph setup",
            "source": "https://github.com/yh-yao/FedGCN/blob/master/src/utils.py",
            "adapted_into": "src/federated_gcn_aml/federated/graph_builder.py",
            "reuse": "methodological foundation with implementation-level adaptation",
            "deviation": "AMLSim banks define clients; h2 uses aggregate PX placeholder rows instead of raw foreign feature subgraphs.",
        },
        {
            "concept": "FedGCN server broadcasts and averages client parameters",
            "source": "https://github.com/yh-yao/FedGCN/blob/master/src/server_class.py",
            "adapted_into": "src/federated_gcn_aml/federated/server.py and strategies.py",
            "reuse": "methodological foundation with implementation-level adaptation",
            "deviation": "Binary BCE model, validation AUPR checkpointing, and optional train-node-weighted FedAvg.",
        },
        {
            "concept": "FedGraph pretraining feature aggregation",
            "source": "https://github.com/FedGraph/fedgraph/blob/main/fedgraph/federated_methods.py",
            "adapted_into": "src/federated_gcn_aml/federated/protocols.py",
            "reuse": "methodological foundation with protocol-boundary adaptation",
            "deviation": "Uses PyG GCN normalization to match centralized GCNConv semantics; local simulation supports plain and HE pretraining transport.",
        },
        {
            "concept": "AggreGCN first trainable layer is linear after feature aggregation",
            "source": "https://github.com/FedGraph/fedgraph/blob/main/fedgraph/gnn_models.py",
            "adapted_into": "src/federated_gcn_aml/models/gcn.py::PreAggregatedGCN",
            "reuse": "methodological foundation with model-level adaptation",
            "deviation": "Binary logits with BCEWithLogitsLoss instead of multiclass log-softmax.",
        },
    ]


MODEL_UPDATE_COMMUNICATION_ESTIMATE_SCOPE = (
    "implemented_local_protocol_payloads_only_no_key_exchange_no_dropout_recovery_no_secret_sharing_no_finite_field_quantization"
)


def _model_state_size_estimates(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    floating_numel = 0
    state_numel = 0
    floating_bytes = 0
    non_floating_bytes = 0
    for tensor in state_dict.values():
        detached = tensor.detach().cpu()
        numel = int(detached.numel())
        bytes_ = numel * int(detached.element_size())
        state_numel += numel
        if detached.is_floating_point():
            floating_numel += numel
            floating_bytes += bytes_
        else:
            non_floating_bytes += bytes_
    return {
        "floating_numel": floating_numel,
        "state_numel": state_numel,
        "floating_bytes": floating_bytes,
        "non_floating_bytes": non_floating_bytes,
        "state_bytes": floating_bytes + non_floating_bytes,
    }


def _model_update_communication_estimates(args, client_graphs, result) -> dict:
    sizes = _model_state_size_estimates(result.model_state)
    rounds = int(args.rounds)
    valid_clients = sum(1 for client in client_graphs if int(client.diagnostics.train_nodes) > 0)
    client_count = len(client_graphs)
    plain_per_client = int(sizes["state_bytes"])
    plain_total_upload = plain_per_client * valid_clients * rounds
    broadcast_per_client = int(sizes["state_bytes"])
    broadcast_total = broadcast_per_client * client_count * rounds

    model_update_diagnostics = result.diagnostics.get("model_update_aggregation", {})
    masked_vector_size = model_update_diagnostics.get("masked_fedavg_vector_numel")
    masked_per_client = None
    masked_total_upload = None
    if args.model_update_aggregation == "masked_fedavg":
        vector_size = int(masked_vector_size if masked_vector_size is not None else sizes["floating_numel"])
        masked_vector_size = vector_size
        masked_per_client = vector_size * 8 + int(sizes["non_floating_bytes"])
        masked_total_upload = masked_per_client * valid_clients * rounds

    return {
        "model_update_num_parameters": int(sizes["floating_numel"]),
        "model_update_state_numel": int(sizes["state_numel"]),
        "model_update_state_bytes_plain_estimate": int(sizes["state_bytes"]),
        "model_update_non_floating_state_bytes_estimate": int(sizes["non_floating_bytes"]),
        "model_update_bytes_per_client_per_round_plain_estimate": plain_per_client,
        "model_update_total_upload_bytes_plain_estimate": plain_total_upload,
        "model_update_global_broadcast_bytes_per_client_per_round_estimate": broadcast_per_client,
        "model_update_total_global_broadcast_download_bytes_estimate": broadcast_total,
        "model_update_total_rounds": rounds,
        "masked_fedavg_vector_size": masked_vector_size,
        "model_update_bytes_per_client_per_round_masked_estimate": masked_per_client,
        "model_update_total_upload_bytes_masked_estimate": masked_total_upload,
        "model_update_communication_estimate_scope": MODEL_UPDATE_COMMUNICATION_ESTIMATE_SCOPE,
    }


def _server_visible_plaintext_feature_contributions(args) -> bool | None:
    if args.federated_mode != "fedgcn_feature_pretrain":
        return None
    return args.feature_pretrain_transport == "plain"


def _make_metadata(args, context, graph, data, scaling, partition, client_graphs, result, model_class_name, runtime=None):
    communication_diagnostics = result.diagnostics.get("pretrain_protocol", {})
    communication_stats = communication_diagnostics.get("communication") or {}
    communication_stats_metadata = communication_stats.get("metadata") or {}
    model_update_diagnostics = result.diagnostics.get("model_update_aggregation", {})
    model_update_communication = _model_update_communication_estimates(args, client_graphs, result)
    runtime = dict(runtime or {})
    payload_policy = (
        args.feature_pretrain_payload_protection if args.federated_mode == "fedgcn_feature_pretrain" else "none"
    )
    payload_protection_enabled = payload_policy != "none"
    pretrain_he_enabled = args.federated_mode == "fedgcn_feature_pretrain" and args.feature_pretrain_transport == "he"
    pretrain_privacy_enabled = args.federated_mode == "fedgcn_feature_pretrain" and (
        pretrain_he_enabled or payload_protection_enabled
    )
    model_update_secure = args.model_update_aggregation == "masked_fedavg"
    privacy_scopes = []
    if pretrain_he_enabled and payload_protection_enabled:
        privacy_scopes.append("pretraining_transport_and_payload_protection")
    elif pretrain_he_enabled:
        privacy_scopes.append("pretraining_transport_only")
    elif payload_protection_enabled:
        privacy_scopes.append("pretraining_payload_protection")
    if model_update_secure:
        privacy_scopes.append("model_update_secure_aggregation")
    total_placeholder_rows = sum(int(client.diagnostics.placeholder_context_rows) for client in client_graphs)
    total_cross_edge_references = sum(int(client.diagnostics.cross_edge_references) for client in client_graphs)
    return {
        "architecture": "federated",
        "experiment_type": "federated_amlsim_gcn",
        "legacy_scaffold_preserved": True,
        "privacy_enforced": bool(pretrain_privacy_enabled or model_update_secure),
        "privacy_scope": "none" if not privacy_scopes else "+".join(privacy_scopes),
        "protocol_simulation": True,
        "secure_aggregation": bool(model_update_diagnostics.get("secure_aggregation", model_update_secure)),
        "secure_aggregation_protocol": model_update_diagnostics.get("secure_aggregation_protocol"),
        "secure_aggregation_claim": model_update_diagnostics.get("secure_aggregation_claim"),
        "secure_aggregation_min_clients": model_update_diagnostics.get("secure_aggregation_min_clients"),
        "secure_aggregation_seed": args.masked_fedavg_seed if model_update_secure else None,
        "secure_aggregation_dropout_resilience": model_update_diagnostics.get(
            "secure_aggregation_dropout_resilience"
        ),
        "secure_aggregation_secret_sharing": model_update_diagnostics.get("secure_aggregation_secret_sharing"),
        "secure_aggregation_finite_field_quantization": model_update_diagnostics.get(
            "secure_aggregation_finite_field_quantization"
        ),
        "secure_aggregation_malicious_server_protection": model_update_diagnostics.get(
            "secure_aggregation_malicious_server_protection"
        ),
        "secure_aggregation_client_collusion_protection": model_update_diagnostics.get(
            "secure_aggregation_client_collusion_protection"
        ),
        "homomorphic_encryption": args.federated_mode == "fedgcn_feature_pretrain" and args.feature_pretrain_transport == "he",
        "feature_pretrain_transport": args.feature_pretrain_transport if args.federated_mode == "fedgcn_feature_pretrain" else "none",
        "feature_pretrain_payload_protection": (
            args.feature_pretrain_payload_protection if args.federated_mode == "fedgcn_feature_pretrain" else "none"
        ),
        "feature_pretrain_payload_protection_requested": (
            getattr(args, "feature_pretrain_payload_protection_requested", args.feature_pretrain_payload_protection)
            if args.federated_mode == "fedgcn_feature_pretrain"
            else "none"
        ),
        "payload_filtering": False,
        "differential_privacy": bool(communication_stats_metadata.get("differential_privacy", False)),
        "formal_differential_privacy": bool(communication_stats_metadata.get("formal_differential_privacy", False)),
        "formal_dp_accounting": communication_stats_metadata.get("formal_dp_accounting", False),
        "diagnostic_perturbation": bool(communication_stats_metadata.get("diagnostic_perturbation", False)),
        "diagnostic_perturbation_mechanism": communication_stats_metadata.get(
            "diagnostic_perturbation_mechanism"
        ),
        "diagnostic_perturbation_scope": communication_stats_metadata.get("diagnostic_perturbation_scope"),
        "diagnostic_noise_multiplier": communication_stats_metadata.get("diagnostic_noise_multiplier"),
        "diagnostic_noise_std": communication_stats_metadata.get("diagnostic_noise_std"),
        "diagnostic_seed": communication_stats_metadata.get("diagnostic_seed", args.diagnostic_seed),
        "payload_protection_kind": communication_stats_metadata.get("payload_protection_kind"),
        "payload_clip_norm": communication_stats_metadata.get("payload_clip_norm"),
        "payload_dp_claim": communication_stats_metadata.get("payload_dp_claim"),
        "claim_scope": communication_stats_metadata.get("claim_scope"),
        "privacy_unit": communication_stats_metadata.get("privacy_unit"),
        "adjacency": communication_stats_metadata.get("adjacency"),
        "payload_dp_mechanism": communication_stats_metadata.get("payload_dp_mechanism"),
        "payload_dp_scope": communication_stats_metadata.get("payload_dp_scope"),
        "payload_dp_clip_norm": communication_stats_metadata.get("payload_dp_clip_norm"),
        "payload_dp_sensitivity_l2": communication_stats_metadata.get("payload_dp_sensitivity_l2"),
        "payload_dp_noise_multiplier": communication_stats_metadata.get("payload_dp_noise_multiplier"),
        "payload_dp_noise_std": communication_stats_metadata.get("payload_dp_noise_std"),
        "payload_dp_seed": communication_stats_metadata.get("payload_dp_seed", args.payload_dp_seed),
        "payload_dp_epsilon": communication_stats_metadata.get("payload_dp_epsilon"),
        "payload_dp_delta": communication_stats_metadata.get("payload_dp_delta"),
        "payload_dp_calibration": communication_stats_metadata.get("payload_dp_calibration"),
        "dp_epsilon": communication_stats_metadata.get("dp_epsilon"),
        "dp_delta": communication_stats_metadata.get("dp_delta"),
        "epsilon_per_release": communication_stats_metadata.get("epsilon_per_release"),
        "delta_per_release": communication_stats_metadata.get("delta_per_release"),
        "epsilon_total": communication_stats_metadata.get("epsilon_total"),
        "delta_total": communication_stats_metadata.get("delta_total"),
        "composition": communication_stats_metadata.get("composition"),
        "num_noised_releases": communication_stats_metadata.get("num_noised_releases"),
        "num_unique_private_units": communication_stats_metadata.get("num_unique_private_units"),
        "max_release_count_per_unit": communication_stats_metadata.get("max_release_count_per_unit"),
        "duplicate_release_policy": communication_stats_metadata.get("duplicate_release_policy"),
        "trust_boundary": communication_stats_metadata.get("trust_boundary"),
        "not_claimed": communication_stats_metadata.get("not_claimed"),
        "formal_dp_oracle_diagnostics_enabled": communication_stats_metadata.get("formal_dp_oracle_diagnostics_enabled"),
        "diagnostics_scope": communication_stats_metadata.get("diagnostics_scope"),
        "rows_clipped": communication_stats_metadata.get("rows_clipped"),
        "rows_noised": communication_stats_metadata.get("rows_noised"),
        "clip_fraction": communication_stats_metadata.get("clip_fraction"),
        "payload_protection_policy": payload_policy,
        "server_can_decrypt": False if args.federated_mode == "fedgcn_feature_pretrain" and args.feature_pretrain_transport == "he" else None,
        "server_can_inspect_pretrain_plaintext_values": (
            args.federated_mode == "fedgcn_feature_pretrain" and args.feature_pretrain_transport == "plain"
        ),
        "model_updates_encrypted": False,
        "model_updates_visible_to_server": bool(
            model_update_diagnostics.get("model_updates_visible_to_server", not model_update_secure)
        ),
        "plaintext_model_updates_visible_to_server": bool(
            model_update_diagnostics.get("plaintext_model_updates_visible_to_server", not model_update_secure)
        ),
        "masked_model_updates_visible_to_server": bool(
            model_update_diagnostics.get("masked_model_updates_visible_to_server", model_update_secure)
        ),
        "aggregate_update_visible_to_server": bool(
            model_update_diagnostics.get("aggregate_update_visible_to_server", True)
        ),
        "model_update_dp": bool(model_update_diagnostics.get("model_update_dp", False)),
        "model_update_aggregation": args.model_update_aggregation,
        **model_update_communication,
        "global_preprocessing_centralized": True,
        "topology_privacy": False,
        "central_px_shortcut": False,
        "server_visible_plaintext_feature_contributions": _server_visible_plaintext_feature_contributions(args),
        "contribution_locality": "owned_features_only" if args.federated_mode == "fedgcn_feature_pretrain" else "no_feature_contribution",
        "client_receives_full_norm_operator_for_contribution": (
            False if args.federated_mode == "fedgcn_feature_pretrain" else None
        ),
        "client_visible_propagation_weights": args.federated_mode == "fedgcn_feature_pretrain",
        "client_visible_cross_client_norm_weights": args.federated_mode == "fedgcn_feature_pretrain",
        "topology_private_against_clients": False if args.federated_mode == "fedgcn_feature_pretrain" else None,
        "client_norm_operator_filter": (
            "source_owned_edges_to_needed_target_rows" if args.federated_mode == "fedgcn_feature_pretrain" else "none"
        ),
        "data_root": str(context.data_root),
        "output_dir": str(context.output_dir),
        "representation_version": graph.representation_version,
        "feature_boundary": graph.feature_boundary,
        "configured_removed_feature_columns": list(graph.configured_removed_feature_columns),
        "actually_removed_feature_columns": list(graph.actually_removed_feature_columns),
        "feature_columns_before_boundary": list(graph.feature_columns_before_boundary),
        "feature_columns_after_boundary": list(graph.feature_columns_after_boundary),
        "feature_count_before_boundary": graph.feature_count_before_boundary,
        "feature_count_after_boundary": graph.feature_count_after_boundary,
        "feature_boundary_assumption": graph.feature_boundary_assumption,
        "federated_mode": args.federated_mode,
        "num_hops": args.num_hops,
        "mode_semantics": (
            "cut_edges owned-only intra-client graph"
            if args.federated_mode == "cut_edges"
            else "FedGCN-style pretraining payload installs PX rows; h2 is not the old cumulative scaffold h2 ablation"
        ),
        "raw_foreign_features_exposed": False,
        "raw_foreign_labels_exposed": False,
        "placeholder_rows_used": total_placeholder_rows > 0,
        "total_placeholder_rows": int(total_placeholder_rows),
        "cross_edge_references_used": total_cross_edge_references > 0,
        "num_cross_edge_references": int(total_cross_edge_references),
        "communication_only": args.federated_mode == "fedgcn_feature_pretrain",
        "pretrain_protocol": communication_diagnostics,
        "model_update_aggregation_diagnostics": model_update_diagnostics,
        "feature_operator": "none" if args.federated_mode == "cut_edges" else "P X",
        "normalization": "none" if args.federated_mode == "cut_edges" else "PyG gcn_norm",
        "self_loops": None if args.federated_mode == "cut_edges" else True,
        "directionality": None if args.federated_mode == "cut_edges" else "source_to_target",
        "edge_weight_mode": args.edge_weight_mode,
        "aggregation_weighting": args.aggregation_weighting,
        "masked_fedavg_rounds": model_update_diagnostics.get("masked_fedavg_rounds"),
        "masked_fedavg_valid_clients_min": model_update_diagnostics.get("masked_fedavg_valid_clients_min"),
        "masked_fedavg_valid_clients_max": model_update_diagnostics.get("masked_fedavg_valid_clients_max"),
        "masked_fedavg_vector_numel": model_update_diagnostics.get("masked_fedavg_vector_numel"),
        "masked_fedavg_missing_client_policy": model_update_diagnostics.get("masked_fedavg_missing_client_policy"),
        "masked_fedavg_non_floating_state_policy": model_update_diagnostics.get(
            "masked_fedavg_non_floating_state_policy"
        ),
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "optimizer_state_policy": "reset_per_round",
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "model_depth": args.num_layers,
        "h2_exact_for_model_depth": (
            True
            if args.federated_mode == "fedgcn_feature_pretrain" and args.num_hops == 2 and args.num_layers == 2
            else "partial"
            if args.federated_mode == "fedgcn_feature_pretrain" and args.num_hops == 2
            else None
        ),
        "h2_architecture": (
            "preaggregated_linear_first"
            if args.federated_mode == "fedgcn_feature_pretrain" and args.num_hops == 2
            else None
        ),
        "preaggregated_architecture": (
            "preaggregated_linear_first" if args.federated_mode == "fedgcn_feature_pretrain" else None
        ),
        "legacy_px_local_gcn": False if args.federated_mode == "fedgcn_feature_pretrain" and args.num_hops == 2 else None,
        "fedgraph_aggre_gcn_style": args.federated_mode == "fedgcn_feature_pretrain",
        "model_class": model_class_name,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_node_features": data.num_node_features,
        "positive_labels": int((data.y == 1).sum().item()),
        "split_counts": split_counts(data),
        "expected_val_count": int(data.val_mask.sum().item()),
        "actual_val_count": int(result.metrics["actual_val_count"]),
        "expected_test_count": int(data.test_mask.sum().item()),
        "actual_test_count": int(result.metrics["actual_test_count"]),
        "graph_semantics": graph.graph_semantics,
        "feature_names": list(graph.feature_columns),
        "feature_groups": graph.feature_groups,
        "feature_semantics": graph.feature_semantics,
        "log1p_columns": list(graph.log1p_columns),
        "scaling": scaling.metadata,
        "partition": partition.diagnostics,
        "client_diagnostics": [client.diagnostics.__dict__ for client in client_graphs],
        "communication_diagnostics": communication_diagnostics,
        "uploaded_rows": communication_stats.get("upload_rows"),
        "uploaded_row_occurrences": communication_stats_metadata.get("uploaded_row_occurrences"),
        "unique_uploaded_source_target_rows": communication_stats_metadata.get("unique_uploaded_source_target_rows"),
        "duplicate_upload_factor": communication_stats_metadata.get("duplicate_upload_factor"),
        "duplicate_placeholder_rows_h2": communication_stats_metadata.get("duplicate_placeholder_rows_h2"),
        "downloaded_rows": communication_stats.get("download_rows"),
        "uploaded_plain_elements": communication_stats_metadata.get("uploaded_plain_elements"),
        "downloaded_plain_elements": communication_stats_metadata.get("downloaded_plain_elements"),
        "payload_tensor_elements": communication_stats_metadata.get("payload_tensor_elements"),
        "approx_payload_bytes_float32": communication_stats_metadata.get("approx_payload_bytes_float32"),
        "uploaded_blocks": communication_stats_metadata.get("uploaded_blocks"),
        "downloaded_blocks": communication_stats_metadata.get("downloaded_blocks"),
        "uploaded_ciphertext_blocks": communication_stats_metadata.get("uploaded_ciphertext_blocks"),
        "downloaded_ciphertext_blocks": communication_stats_metadata.get("downloaded_ciphertext_blocks"),
        "uploaded_ciphertext_bytes": communication_stats_metadata.get("uploaded_ciphertext_bytes"),
        "downloaded_ciphertext_bytes": communication_stats_metadata.get("downloaded_ciphertext_bytes"),
        "ciphertext_bytes_per_plain_element": communication_stats_metadata.get("ciphertext_bytes_per_plain_element"),
        "ckks_payload_max_abs_error": communication_stats_metadata.get("ckks_payload_max_abs_error"),
        "ckks_payload_mean_abs_error": communication_stats_metadata.get("ckks_payload_mean_abs_error"),
        "ckks_payload_error_sampled": communication_stats_metadata.get("ckks_payload_error_sampled"),
        "feature_pretrain_block_rows": args.feature_pretrain_block_rows,
        "pretrain_timing": communication_stats_metadata.get("timing"),
        "total_pretraining_protocol_seconds": communication_diagnostics.get("total_pretraining_protocol_seconds"),
        "total_run_wall_time_sec": runtime.get("total_run_wall_time_sec"),
        "training_wall_time_sec": runtime.get("training_wall_time_sec"),
        "pretraining_wall_time_sec": runtime.get(
            "pretraining_wall_time_sec",
            communication_diagnostics.get("total_pretraining_protocol_seconds"),
        ),
        "best_round_or_epoch": result.metrics.get("best_round"),
        "final_round_or_epoch": args.rounds,
        "runtime_measurement_scope": "single_process_runner_wall_time_hardware_dependent",
        "external_reference_mapping": external_reference_mapping(),
        "metrics": result.metrics,
        "round_metrics_path": "round_metrics.csv",
        "client_summary_path": "client_summary.json",
    }


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    total_run_t0 = time.perf_counter()
    args = parse_args(argv)
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
    logger.info(
        "Constructed AMLSim representation=%s feature_boundary=%s feature_dim_before=%d "
        "feature_dim_after=%d removed_features=%s",
        graph.representation_version,
        graph.feature_boundary,
        graph.feature_count_before_boundary,
        graph.feature_count_after_boundary,
        list(graph.actually_removed_feature_columns),
    )
    masks = create_amlsim_node_masks(data, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    data = attach_node_masks(data, masks)
    data, scaling = standardize_node_features_train_only(data, graph.feature_columns)

    partition = build_bank_partition(raw.accounts, graph.account_ids, graph.id2idx, y=data.y, masks=_split_masks_dict(data))
    client_graphs = build_client_graphs(
        global_data=data,
        partition=partition,
        mode=args.federated_mode,
        num_hops=args.num_hops,
        edge_weight_mode=args.edge_weight_mode,
    )

    device = resolve_device(args.device)
    model_config = {
        "in_channels": data.num_node_features,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model_cls = select_model_class(args.federated_mode, args.num_hops)
    initial_model = model_cls(**model_config)
    initial_state = {key: value.detach().cpu().clone() for key, value in initial_model.state_dict().items()}
    clients = [
        FederatedClient(
            client_id=client_graph.client_id,
            data=client_graph.data,
            model_config=model_config,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            model_cls=model_cls,
            masked_fedavg_seed=args.masked_fedavg_seed if args.model_update_aggregation == "masked_fedavg" else None,
        )
        for client_graph in client_graphs
    ]
    protocol = (
        NoCommunicationProtocol()
        if args.federated_mode == "cut_edges"
        else FedGCNFeaturePretrainProtocol(
            num_hops=args.num_hops,
            edge_weight_mode=args.edge_weight_mode,
            feature_pretrain_transport=args.feature_pretrain_transport,
            feature_pretrain_payload_protection=args.feature_pretrain_payload_protection,
            payload_clip_norm=args.payload_clip_norm,
            diagnostic_noise_multiplier=args.diagnostic_noise_multiplier,
            diagnostic_seed=args.diagnostic_seed,
            payload_dp_epsilon=args.payload_dp_epsilon,
            payload_dp_delta=args.payload_dp_delta,
            payload_dp_calibration=args.payload_dp_calibration,
            payload_dp_seed=args.payload_dp_seed,
            block_rows=args.feature_pretrain_block_rows,
        )
    )
    protocol.prepare(data, partition, client_graphs)
    evaluator = GlobalOwnedNodeEvaluator(int(data.val_mask.sum().item()), int(data.test_mask.sum().item()))
    server = FederatedServer(
        initial_state,
        clients,
        evaluator,
        FedAvgStrategy(
            weighting=args.aggregation_weighting,
            model_update_aggregation=args.model_update_aggregation,
            masked_fedavg_min_clients=args.masked_fedavg_min_clients,
        ),
        protocol,
    )
    if args.federated_mode == "fedgcn_feature_pretrain":
        server.run_pretraining_protocol()
    else:
        server.pretrain_diagnostics = protocol.diagnostics()

    logger.info(
        "Starting federated AMLSim run: mode=%s hops=%d clients=%d rounds=%d local_epochs=%d",
        args.federated_mode,
        args.num_hops,
        len(clients),
        args.rounds,
        args.local_epochs,
    )
    training_t0 = time.perf_counter()
    result = server.run(rounds=args.rounds, local_epochs=args.local_epochs)
    training_wall_time_sec = time.perf_counter() - training_t0

    torch.save(result.model_state, context.output_dir / "model.pt")
    if args.save_predictions:
        torch.save(result.predictions, context.output_dir / "best_predictions.pt")
    _write_round_metrics(context.output_dir / "round_metrics.csv", result.round_metrics)

    client_summary = {
        "partition": partition.diagnostics,
        "client_graphs": {client.client_id: client.diagnostics.__dict__ for client in client_graphs},
        "clients": result.diagnostics.get("clients", []),
    }
    with (context.output_dir / "client_summary.json").open("w") as f:
        json.dump(client_summary, f, indent=2)

    metadata = _make_metadata(
        args,
        context,
        graph,
        data,
        scaling,
        partition,
        client_graphs,
        result,
        model_cls.__name__,
        runtime={
            "total_run_wall_time_sec": time.perf_counter() - total_run_t0,
            "training_wall_time_sec": training_wall_time_sec,
            "pretraining_wall_time_sec": server.pretrain_diagnostics.get("total_pretraining_protocol_seconds"),
        },
    )
    with (context.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved federated AMLSim outputs to %s; best round=%d val AUPR=%.4f test AUPR=%.4f",
        context.output_dir,
        result.metrics["best_round"],
        result.metrics["val"]["AUPR"],
        result.metrics["test"]["AUPR"],
    )


if __name__ == "__main__":
    main()
