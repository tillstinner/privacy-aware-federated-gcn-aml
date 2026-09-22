from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class RecipientRowPlan:
    """Rows a recipient client is allowed to receive for feature pretraining."""

    client_id: str
    global_node_ids: torch.Tensor
    owned_rows: torch.Tensor
    placeholder_rows: torch.Tensor
    mode: Literal["h1", "h2"]


@dataclass(frozen=True)
class BlockPlan:
    """Recipient-aligned row block for routed pretraining payloads."""

    recipient_client_id: str
    block_id: str
    row_ids: torch.Tensor
    shape: tuple[int, int]
    row_start: int
    row_end: int


@dataclass(frozen=True)
class FeatureContributionBlock:
    """Plain contribution block before transport encoding.

    The tensor is computed inside the source client from owned raw features and
    sparse normalized edge filtering; it is not a server-side dense P[B,V] slice.
    """

    source_client_id: str
    recipient_client_id: str
    block_id: str
    row_ids: torch.Tensor
    contribution: torch.Tensor
    feature_shape: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedContributionBlock:
    """Transport-encoded source-to-server pretraining contribution block."""

    source_client_id: str
    recipient_client_id: str
    block_id: str
    row_ids: torch.Tensor
    payload: Any
    feature_shape: tuple[int, int]
    transport: str
    serialized_bytes: int = 0
    is_encrypted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedPayloadBlock:
    """Transport-encoded server-to-client aggregate payload block."""

    recipient_client_id: str
    block_id: str
    row_ids: torch.Tensor
    payload: Any
    feature_shape: tuple[int, int]
    transport: str
    serialized_bytes: int = 0
    is_encrypted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommunicationStats:
    """Local simulation communication metadata, not encrypted network telemetry."""

    protocol: str
    num_hops: int
    upload_rows: int = 0
    download_rows: int = 0
    feature_dim: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientDiagnostics:
    client_id: str
    owned_nodes: int
    train_nodes: int
    val_nodes: int
    test_nodes: int
    positive_train_labels: int
    positive_val_labels: int
    positive_test_labels: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientGraphDiagnostics:
    client_id: str
    owned_nodes: int
    compute_rows: int
    placeholder_context_rows: int
    raw_foreign_features_exposed: bool
    edges: int
    owned_internal_edges: int
    cross_edge_references: int
    train_nodes: int
    val_nodes: int
    test_nodes: int
    positive_train_labels: int
    positive_val_labels: int
    positive_test_labels: int
    pretrain_payload_rows: int = 0
    pretrain_feature_source: str = "none"
    communication_only: bool = False


@dataclass(frozen=True)
class ClientFeatureContribution:
    """Client-to-server aggregate feature contribution for local protocol simulation.

    This legacy aggregate message remains for compatibility with plaintext tests.
    New privacy-aware pretraining uses recipient-aligned contribution blocks and
    transport-encoded messages.
    """

    client_id: str
    target_global_node_ids: torch.Tensor
    contribution: torch.Tensor
    feature_shape: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServerFeaturePayload:
    """Server-to-client pretraining payload for local protocol simulation.

    The rows contain aggregated features such as PX, never raw foreign features
    or labels.
    """

    client_id: str
    global_node_ids: torch.Tensor
    aggregated_features: torch.Tensor
    num_hops: int
    feature_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    state_dict: dict[str, torch.Tensor]
    train_node_count: int
    loss: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelTensorSpec:
    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int


@dataclass(frozen=True)
class ModelUpdateAggregationPlan:
    round_idx: int
    mode: Literal["plain_fedavg", "masked_fedavg"]
    participant_client_ids: tuple[str, ...]
    weights: dict[str, float]
    tensor_specs: tuple[ModelTensorSpec, ...]
    vector_numel: int
    masked_fedavg_min_clients: int = 2


@dataclass(frozen=True)
class ClientModelUpdateMessage:
    client_id: str
    train_node_count: int
    loss: float | None
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    masked_update_vector: torch.Tensor | None = None
    non_floating_state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionSplitPacket:
    split: str
    global_node_indices: torch.Tensor
    y: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class ClientPredictionPacket:
    client_id: str
    val: PredictionSplitPacket
    test: PredictionSplitPacket


@dataclass(frozen=True)
class RoundMetrics:
    round: int
    mean_train_loss: float | None
    val_AUPR: float
    val_AUROC: float | None
    val_F1: float
    val_MCC: float
    val_precision: float
    val_recall: float
    val_TP: int
    val_FP: int
    val_TN: int
    val_FN: int
    test_AUPR: float
    test_AUROC: float | None
    test_F1: float
    test_MCC: float
    test_precision: float
    test_recall: float
    test_TP: int
    test_FP: int
    test_TN: int
    test_FN: int
    selected_threshold: float
