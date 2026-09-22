from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from federated_gcn_aml.federated.messages import (
    BlockPlan,
    ClientFeatureContribution,
    CommunicationStats,
    EncodedContributionBlock,
    EncodedPayloadBlock,
    FeatureContributionBlock,
    RecipientRowPlan,
    ServerFeaturePayload,
)
from federated_gcn_aml.federated.payload_protection import PayloadProtectionPolicy, make_payload_protection_policy
from federated_gcn_aml.federated.transports import FeaturePretrainTransport, make_feature_pretrain_transport


@dataclass
class BasePretrainProtocol:
    name: str
    num_hops: int = 0
    edge_weight_mode: str = "unit"
    privacy_enforced: bool = False
    secure_aggregation: bool = False
    homomorphic_encryption: bool = False
    simulation_type: str = "local_protocol_simulation"
    raw_foreign_features_exposed: bool = False

    def prepare(self, global_data, partition, client_graphs) -> None:
        self.global_data = global_data
        self.partition = partition
        self.client_graphs = {graph.client_id: graph for graph in client_graphs}

    def collect_contributions(self, clients) -> list[Any]:
        return []

    def aggregate_on_server(self, contributions: list[Any]) -> None:
        return None

    def build_payload_for_client(self, client_id: str) -> ServerFeaturePayload:
        graph = self.client_graphs[client_id]
        return ServerFeaturePayload(
            client_id=client_id,
            global_node_ids=graph.global_node_indices.clone(),
            aggregated_features=graph.data.x.detach().cpu().clone(),
            num_hops=self.num_hops,
            feature_name="raw_owned_features",
        )

    def install_payload(self, client, payload) -> None:
        client.install_feature_payload(payload)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": self.name,
            "num_hops": self.num_hops,
            "privacy_enforced": self.privacy_enforced,
            "secure_aggregation": self.secure_aggregation,
            "homomorphic_encryption": self.homomorphic_encryption,
            "simulation_type": self.simulation_type,
            "raw_foreign_features_exposed": self.raw_foreign_features_exposed,
            "central_px_shortcut": False,
            "contribution_locality": "no_feature_contribution",
            "raw_foreign_labels_exposed": False,
        }


class NoCommunicationProtocol(BasePretrainProtocol):
    def __init__(self) -> None:
        super().__init__(name="no_communication", num_hops=0)


@dataclass
class FedGCNFeaturePretrainProtocol(BasePretrainProtocol):
    num_hops: int = 1
    edge_weight_mode: str = "unit"
    include_self: bool = True
    flow: str = "source_to_target"
    feature_pretrain_transport: str = "plain"
    feature_pretrain_payload_protection: str = "none"
    payload_clip_norm: float | None = None
    diagnostic_noise_multiplier: float | None = None
    diagnostic_seed: int | None = None
    payload_dp_epsilon: float | None = None
    payload_dp_delta: float | None = None
    payload_dp_calibration: str = "classical_gaussian"
    payload_dp_seed: int | None = None
    block_rows: int = 64
    name: str = "fedgcn_feature_pretrain"
    transport: FeaturePretrainTransport | None = None
    payload_protection: PayloadProtectionPolicy | None = None
    _norm_edge_index: torch.Tensor | None = field(default=None, init=False, repr=False)
    _norm_edge_weight: torch.Tensor | None = field(default=None, init=False, repr=False)
    _recipient_plans: dict[str, RecipientRowPlan] = field(default_factory=dict, init=False, repr=False)
    _block_plans: list[BlockPlan] = field(default_factory=list, init=False, repr=False)
    _block_plan_by_id: dict[tuple[str, str], BlockPlan] = field(default_factory=dict, init=False, repr=False)
    _needed_target_mask: torch.Tensor | None = field(default=None, init=False, repr=False)
    _payload_blocks_by_client: dict[str, list[EncodedPayloadBlock]] = field(default_factory=dict, init=False, repr=False)
    _plaintext_payload_oracle_by_key: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _stats: CommunicationStats | None = field(default=None, init=False, repr=False)
    _timings: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_hops not in {1, 2}:
            raise ValueError("FedGCNFeaturePretrainProtocol supports num_hops=1 or num_hops=2")
        if self.block_rows <= 0:
            raise ValueError("block_rows must be positive")
        self.transport = self.transport or make_feature_pretrain_transport(self.feature_pretrain_transport)
        self.payload_protection = self.payload_protection or make_payload_protection_policy(
            self.feature_pretrain_payload_protection,
            clip_norm=self.payload_clip_norm,
            diagnostic_noise_multiplier=self.diagnostic_noise_multiplier,
            diagnostic_seed=self.diagnostic_seed,
            payload_dp_seed=self.payload_dp_seed,
            epsilon=self.payload_dp_epsilon,
            delta=self.payload_dp_delta,
            calibration=self.payload_dp_calibration,
        )
        self.feature_pretrain_transport = self.transport.name
        self.feature_pretrain_payload_protection = self.payload_protection.name
        self.privacy_enforced = self.transport.is_encrypted
        self.secure_aggregation = False
        self.homomorphic_encryption = self.transport.is_encrypted
        self.simulation_type = "local_protocol_simulation"
        self.raw_foreign_features_exposed = False

    def prepare(self, global_data, partition, client_graphs) -> None:
        t0 = time.perf_counter()
        super().prepare(global_data, partition, client_graphs)
        edge_weight = None
        if self.edge_weight_mode == "original":
            edge_weight = getattr(global_data, "edge_weight", None)
        elif self.edge_weight_mode != "unit":
            raise ValueError(f"Unsupported edge_weight_mode: {self.edge_weight_mode}")
        self._norm_edge_index, self._norm_edge_weight = gcn_norm(
            global_data.edge_index.cpu(),
            edge_weight=None if edge_weight is None else edge_weight.cpu().float(),
            num_nodes=global_data.num_nodes,
            improved=False,
            add_self_loops=self.include_self,
            flow=self.flow,
            dtype=global_data.x.dtype,
        )
        t_after_norm = time.perf_counter()
        self._build_recipient_and_block_plans(client_graphs, int(global_data.x.size(1)))
        t_after_plans = time.perf_counter()
        self._timings.update(
            {
                "normalization_seconds": t_after_norm - t0,
                "recipient_row_and_block_planning_seconds": t_after_plans - t_after_norm,
            }
        )

    def _build_recipient_and_block_plans(self, client_graphs, feature_dim: int) -> None:
        self._recipient_plans = {}
        self._block_plans = []
        self._block_plan_by_id = {}
        self._needed_target_mask = torch.zeros(int(self.global_data.num_nodes), dtype=torch.bool)
        mode = "h1" if self.num_hops == 1 else "h2"
        for graph in client_graphs:
            global_node_ids = graph.global_node_indices.detach().cpu().long()
            placeholder_mask = graph.data.placeholder_context_mask.detach().cpu().bool()
            owned_mask = ~placeholder_mask
            row_plan = RecipientRowPlan(
                client_id=graph.client_id,
                global_node_ids=global_node_ids.clone(),
                owned_rows=global_node_ids[owned_mask].clone(),
                placeholder_rows=global_node_ids[placeholder_mask].clone(),
                mode=mode,
            )
            self._recipient_plans[graph.client_id] = row_plan
            for block_idx, start in enumerate(range(0, int(global_node_ids.numel()), self.block_rows)):
                end = min(start + self.block_rows, int(global_node_ids.numel()))
                block = BlockPlan(
                    recipient_client_id=graph.client_id,
                    block_id=f"{graph.client_id}:{block_idx}",
                    row_ids=global_node_ids[start:end].clone(),
                    shape=(end - start, feature_dim),
                    row_start=start,
                    row_end=end,
                )
                self._block_plans.append(block)
                self._block_plan_by_id[(block.recipient_client_id, block.block_id)] = block
                if block.row_ids.numel():
                    self._needed_target_mask[block.row_ids.long()] = True

    def _filtered_norm_edges_for_client(self, client) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only normalized propagation entries needed by one source client.

        The local simulation still uses global routing in the server/protocol role,
        but a source client no longer receives the full normalized sparse operator
        for contribution construction. It receives only entries whose source node is
        owned by that client and whose target row is part of at least one recipient
        payload plan.
        """

        if self._norm_edge_index is None or self._norm_edge_weight is None or self._needed_target_mask is None:
            raise RuntimeError("Protocol must be prepared before filtering normalized edges.")
        src, dst = self._norm_edge_index.cpu().long()
        owned_global = client.data.owned_global_node_indices.detach().cpu().long()
        source_mask = torch.zeros(int(self.global_data.num_nodes), dtype=torch.bool)
        source_mask[owned_global] = True
        edge_mask = source_mask[src] & self._needed_target_mask[dst]
        return self._norm_edge_index[:, edge_mask].clone(), self._norm_edge_weight[edge_mask].clone()

    def _duplicate_placeholder_rows_h2(self) -> int:
        if self.num_hops != 2:
            return 0
        recipient_occurrences: Counter[int] = Counter()
        placeholder_rows: set[int] = set()
        for plan in self._recipient_plans.values():
            recipient_occurrences.update(int(row) for row in plan.global_node_ids.tolist())
            placeholder_rows.update(int(row) for row in plan.placeholder_rows.tolist())
        return sum(1 for row in placeholder_rows if recipient_occurrences[row] > 1)

    def _block_plans_for_target_rows(self, target_global_node_ids: torch.Tensor) -> list[BlockPlan]:
        """Return recipient block plans that intersect the given target rows."""

        target_global_node_ids = target_global_node_ids.detach().cpu().long()
        if not target_global_node_ids.numel():
            return []
        target_mask = torch.zeros(int(self.global_data.num_nodes), dtype=torch.bool)
        target_mask[target_global_node_ids] = True
        return [block_plan for block_plan in self._block_plans if bool(target_mask[block_plan.row_ids.long()].any())]

    def _payload_protection_diagnostics(self) -> dict[str, Any]:
        diagnostics = self.payload_protection.diagnostics()
        if diagnostics.get("formal_differential_privacy"):
            if self.payload_protection.name == "local_row_dp":
                diagnostics["trust_boundary"] = (
                    "local_source_he_transport" if self.transport.is_encrypted else "local_source_plain_transport"
                )
            elif self.payload_protection.name == "central_row_dp":
                diagnostics["trust_boundary"] = (
                    "central_he_public_context"
                    if self.transport.is_encrypted
                    else "central_trusted_coordinator_plain_transport"
                )
        return diagnostics

    def collect_contributions(self, clients) -> list[EncodedContributionBlock]:
        if self._norm_edge_index is None or self._norm_edge_weight is None:
            raise RuntimeError("Protocol must be prepared before collecting contributions.")
        encoded: list[EncodedContributionBlock] = []
        self._plaintext_payload_oracle_by_key = {}
        construction_seconds = 0.0
        encoding_seconds = 0.0
        edges_scanned = 0
        edges_selected = 0
        blocks_constructed = 0
        blocks_encoded = 0
        client_visible_norm_edges_total = 0
        client_visible_norm_edges_max = 0

        for client in clients:
            client_norm_edge_index, client_norm_edge_weight = self._filtered_norm_edges_for_client(client)
            client_visible_edges = int(client_norm_edge_index.size(1))
            client_visible_norm_edges_total += client_visible_edges
            client_visible_norm_edges_max = max(client_visible_norm_edges_max, client_visible_edges)
            client_target_rows = client_norm_edge_index[1].detach().cpu().long().unique(sorted=True)
            client_block_plans = self._block_plans_for_target_rows(client_target_rows)
            t0 = time.perf_counter()
            blocks = client.build_feature_contribution_blocks(
                client_block_plans,
                client_norm_edge_index,
                client_norm_edge_weight,
                payload_protection=self.payload_protection,
            )
            t1 = time.perf_counter()
            construction_seconds += t1 - t0
            edges_scanned += client_visible_edges
            edges_selected += client_visible_edges
            blocks_constructed += len(blocks)
            for block in blocks:
                if self.transport.is_encrypted and self.payload_protection.enable_plaintext_oracle_diagnostics():
                    key = (block.recipient_client_id, block.block_id)
                    oracle = self._plaintext_payload_oracle_by_key.setdefault(
                        key, torch.zeros(block.feature_shape, dtype=torch.float)
                    )
                    oracle += block.contribution.to(oracle.dtype)
                block = self.payload_protection.protect_source_contribution_before_transport(block)
                t2 = time.perf_counter()
                encoded.append(self.transport.encode_contribution_block(block))
                t3 = time.perf_counter()
                encoding_seconds += t3 - t2
                blocks_encoded += 1

        self._timings.update(
            {
                "plaintext_contribution_construction_seconds": construction_seconds,
                "transport_encoding_seconds": encoding_seconds,
                "he_encryption_seconds": encoding_seconds if self.transport.is_encrypted else 0.0,
                "contribution_edges_scanned": edges_scanned,
                "contribution_edges_selected": edges_selected,
                "contribution_blocks_constructed": blocks_constructed,
                "contribution_blocks_encoded": blocks_encoded,
                "global_norm_edges": int(self._norm_edge_index.size(1)),
                "client_visible_norm_edges_total": client_visible_norm_edges_total,
                "client_visible_norm_edges_max": client_visible_norm_edges_max,
            }
        )
        return encoded

    def aggregate_on_server(self, contributions: list[EncodedContributionBlock | ClientFeatureContribution]) -> None:
        # Backward-compatible adapter for older direct tests.
        if contributions and isinstance(contributions[0], ClientFeatureContribution):
            contributions = self._legacy_contributions_to_blocks(contributions)  # type: ignore[assignment]

        grouped: dict[tuple[str, str], list[EncodedContributionBlock]] = defaultdict(list)
        uploaded_rows = 0
        uploaded_plain_elements = 0
        uploaded_serialized_bytes = 0
        uploaded_blocks = 0
        uploaded_row_occurrences = 0
        unique_uploaded_source_target_rows: set[tuple[str, int]] = set()
        for contribution in contributions:  # type: ignore[assignment]
            key = (contribution.recipient_client_id, contribution.block_id)
            grouped[key].append(contribution)
            uploaded_rows += int(contribution.feature_shape[0])
            uploaded_plain_elements += int(contribution.feature_shape[0] * contribution.feature_shape[1])
            uploaded_serialized_bytes += int(contribution.serialized_bytes)
            uploaded_blocks += 1
            for row in contribution.row_ids.tolist():
                uploaded_row_occurrences += 1
                unique_uploaded_source_target_rows.add((contribution.source_client_id, int(row)))

        payload_blocks_by_client: dict[str, list[EncodedPayloadBlock]] = defaultdict(list)
        server_aggregation_seconds = 0.0
        payload_protection_seconds = 0.0
        ckks_error_seconds = 0.0
        ckks_abs_error_sum = 0.0
        ckks_error_count = 0
        ckks_payload_max_abs_error = None
        downloaded_rows = 0
        downloaded_plain_elements = 0
        downloaded_serialized_bytes = 0
        downloaded_blocks = 0
        for block_plan in self._block_plans:
            t0 = time.perf_counter()
            aggregate = self.transport.aggregate_blocks(grouped.get((block_plan.recipient_client_id, block_plan.block_id), []), block_plan)
            t1 = time.perf_counter()
            aggregate = self.payload_protection.protect_aggregate_before_delivery(
                aggregate,
                transport=self.transport,
                block_plan=block_plan,
                metadata={"num_hops": self.num_hops},
            )
            plaintext_delta = self.payload_protection.plaintext_aggregate_delta_for_block(block_plan)
            if plaintext_delta is not None and self.payload_protection.enable_plaintext_oracle_diagnostics():
                key = (block_plan.recipient_client_id, block_plan.block_id)
                oracle = self._plaintext_payload_oracle_by_key.setdefault(
                    key, torch.zeros(block_plan.shape, dtype=torch.float)
                )
                oracle += plaintext_delta.to(oracle.dtype)
            t2 = time.perf_counter()
            if self.transport.is_encrypted and self.payload_protection.enable_plaintext_oracle_diagnostics():
                block_max, block_sum, block_count, block_seconds = self._record_ckks_payload_error_diagnostics(
                    aggregate, block_plan
                )
                if block_count:
                    ckks_payload_max_abs_error = (
                        block_max
                        if ckks_payload_max_abs_error is None
                        else max(float(ckks_payload_max_abs_error), float(block_max))
                    )
                    ckks_abs_error_sum += block_sum
                    ckks_error_count += block_count
                ckks_error_seconds += block_seconds
            payload_blocks_by_client[block_plan.recipient_client_id].append(aggregate)
            server_aggregation_seconds += t1 - t0
            payload_protection_seconds += t2 - t1
            downloaded_rows += int(block_plan.shape[0])
            downloaded_plain_elements += int(block_plan.shape[0] * block_plan.shape[1])
            downloaded_serialized_bytes += int(aggregate.serialized_bytes)
            downloaded_blocks += 1

        self._payload_blocks_by_client = {client_id: blocks for client_id, blocks in payload_blocks_by_client.items()}
        feature_dim = int(self.global_data.x.size(1))
        self._timings.update(
            {
                "server_aggregation_seconds": server_aggregation_seconds,
                "payload_protection_before_delivery_seconds": payload_protection_seconds,
                "ckks_payload_error_seconds": ckks_error_seconds,
            }
        )
        ckks_payload_mean_abs_error = (
            float(ckks_abs_error_sum / ckks_error_count)
            if self.transport.is_encrypted and ckks_error_count
            else None
        )
        unique_uploaded_count = len(unique_uploaded_source_target_rows)
        duplicate_upload_factor = (
            float(uploaded_row_occurrences / unique_uploaded_count) if unique_uploaded_count else 0.0
        )
        stats_metadata = {
            "feature_pretrain_transport": self.transport.name,
            "feature_pretrain_payload_protection": self.payload_protection.name,
            "uploaded_rows": uploaded_rows,
            "uploaded_row_occurrences": uploaded_row_occurrences,
            "unique_uploaded_source_target_rows": unique_uploaded_count,
            "duplicate_upload_factor": duplicate_upload_factor,
            "duplicate_placeholder_rows_h2": self._duplicate_placeholder_rows_h2(),
            "downloaded_rows": downloaded_rows,
            "uploaded_plain_elements": uploaded_plain_elements,
            "downloaded_plain_elements": downloaded_plain_elements,
            "approx_plain_float32_bytes": int((uploaded_plain_elements + downloaded_plain_elements) * 4),
            "uploaded_blocks": uploaded_blocks,
            "downloaded_blocks": downloaded_blocks,
            "uploaded_serialized_bytes": uploaded_serialized_bytes,
            "downloaded_serialized_bytes": downloaded_serialized_bytes,
            "uploaded_ciphertext_blocks": uploaded_blocks if self.transport.is_encrypted else 0,
            "downloaded_ciphertext_blocks": downloaded_blocks if self.transport.is_encrypted else 0,
            "uploaded_ciphertext_bytes": uploaded_serialized_bytes if self.transport.is_encrypted else 0,
            "downloaded_ciphertext_bytes": downloaded_serialized_bytes if self.transport.is_encrypted else 0,
            "ciphertext_bytes_per_plain_element": (
                float((uploaded_serialized_bytes + downloaded_serialized_bytes) / (uploaded_plain_elements + downloaded_plain_elements))
                if self.transport.is_encrypted and (uploaded_plain_elements + downloaded_plain_elements)
                else None
            ),
            "ckks_payload_max_abs_error": ckks_payload_max_abs_error if self.transport.is_encrypted else None,
            "ckks_payload_mean_abs_error": ckks_payload_mean_abs_error,
            "ckks_payload_error_sampled": (
                False
                if self.transport.is_encrypted and self.payload_protection.enable_plaintext_oracle_diagnostics()
                else None
            ),
            "contribution_edges_scanned": self._timings.get("contribution_edges_scanned"),
            "contribution_edges_selected": self._timings.get("contribution_edges_selected"),
            "contribution_blocks_constructed": self._timings.get("contribution_blocks_constructed"),
            "payload_tensor_elements": int(downloaded_plain_elements),
            "approx_payload_bytes_float32": int(downloaded_plain_elements * 4),
            "block_rows": self.block_rows,
            "feature_dim": feature_dim,
            "payload_layout": "recipient_aligned_blocks",
            "recipient_extra_rows_decrypted": 0,
            "feature_operator": "P X",
            "normalization": "torch_geometric.nn.conv.gcn_conv.gcn_norm",
            "self_loops": self.include_self,
            "directionality": self.flow,
            "edge_weight_mode": self.edge_weight_mode,
            "central_px_shortcut": False,
            "contribution_locality": "owned_features_only",
            "client_receives_full_norm_operator_for_contribution": False,
            "client_visible_propagation_weights": True,
            "client_visible_cross_client_norm_weights": True,
            "topology_private_against_clients": False,
            "client_norm_operator_filter": "source_owned_edges_to_needed_target_rows",
            "raw_foreign_features_exposed": False,
            "raw_foreign_labels_exposed": False,
            "dense_p_materialized": False,
            "h2_semantics": "installs PX rows for owned plus one-hop placeholder rows; model uses a linear first layer before local normalized aggregation",
            "timing": dict(self._timings),
        }
        stats_metadata.update(self.transport.diagnostics())
        stats_metadata.update(self._payload_protection_diagnostics())
        self._stats = CommunicationStats(
            protocol=self.name,
            num_hops=self.num_hops,
            upload_rows=uploaded_rows,
            download_rows=downloaded_rows,
            feature_dim=feature_dim,
            metadata=stats_metadata,
        )

    def _record_ckks_payload_error_diagnostics(
        self, aggregate: EncodedPayloadBlock, block_plan: BlockPlan
    ) -> tuple[float | None, float, int, float]:
        """Measure CKKS payload error for local simulation validation only.

        This diagnostic decrypts an aggregate block to compare HE transport output
        against the in-memory plaintext oracle. It is instrumentation for research
        validation in the single-process simulator, not deployed server behavior.
        """

        t0 = time.perf_counter()
        decoded_for_error = self.transport.decode_payload_block(aggregate).float()
        oracle = self._plaintext_payload_oracle_by_key.get((block_plan.recipient_client_id, block_plan.block_id))
        if oracle is None:
            oracle = torch.zeros(block_plan.shape, dtype=decoded_for_error.dtype)
        else:
            oracle = oracle.to(decoded_for_error.dtype)
        abs_error = torch.abs(decoded_for_error - oracle)
        elapsed = time.perf_counter() - t0
        if not abs_error.numel():
            return None, 0.0, 0, elapsed
        return (
            float(abs_error.max().item()),
            float(abs_error.sum().item()),
            int(abs_error.numel()),
            elapsed,
        )

    def _legacy_contributions_to_blocks(self, contributions: list[ClientFeatureContribution]) -> list[EncodedContributionBlock]:
        encoded = []
        for contribution in contributions:
            target_ids = contribution.target_global_node_ids.detach().cpu().long()
            tensor = contribution.contribution.detach().cpu()
            lookup_size = 1
            if target_ids.numel():
                lookup_size = max(lookup_size, int(target_ids.max().item()) + 1)
            for block_plan in self._block_plans:
                if block_plan.row_ids.numel():
                    lookup_size = max(lookup_size, int(block_plan.row_ids.max().item()) + 1)
            target_lookup = torch.full((lookup_size,), -1, dtype=torch.long)
            if target_ids.numel():
                target_lookup[target_ids] = torch.arange(target_ids.numel(), dtype=torch.long)
            for block_plan in self._block_plans:
                block_rows = target_lookup[block_plan.row_ids.long()]
                present = block_rows >= 0
                if not bool(present.any()):
                    continue
                block_tensor = torch.zeros(block_plan.shape, dtype=tensor.dtype)
                block_tensor[present] = tensor[block_rows[present]]
                block = FeatureContributionBlock(
                    source_client_id=contribution.client_id,
                    recipient_client_id=block_plan.recipient_client_id,
                    block_id=block_plan.block_id,
                    row_ids=block_plan.row_ids.clone(),
                    contribution=block_tensor,
                    feature_shape=block_plan.shape,
                    metadata={"adapter": "legacy_client_feature_contribution"},
                )
                encoded.append(self.transport.encode_contribution_block(block))
        return encoded

    def build_payload_for_client(self, client_id: str):
        if not self._payload_blocks_by_client:
            raise RuntimeError("Server aggregation must run before building feature payloads.")
        blocks = self._payload_blocks_by_client.get(client_id)
        if blocks is None:
            raise RuntimeError(f"No payload blocks available for client {client_id}.")
        if self.transport.is_encrypted:
            return list(blocks)
        return self._decode_blocks_to_payload(client_id, blocks)

    def _decode_blocks_to_payload(self, client_id: str, blocks: list[EncodedPayloadBlock]) -> ServerFeaturePayload:
        graph = self.client_graphs[client_id]
        node_ids = graph.global_node_indices.cpu().long()
        features = torch.zeros((node_ids.numel(), int(self.global_data.x.size(1))), dtype=self.global_data.x.dtype)
        decode_seconds = 0.0
        for block in blocks:
            block_plan = self._block_plan_by_id[(block.recipient_client_id, block.block_id)]
            t0 = time.perf_counter()
            decoded = self.transport.decode_payload_block(block).to(features.dtype)
            t1 = time.perf_counter()
            decoded = self.payload_protection.protect_decrypted_payload_after_client_decode(decoded)
            features[block_plan.row_start : block_plan.row_end] = decoded
            decode_seconds += t1 - t0
        self._timings["transport_decoding_seconds"] = self._timings.get("transport_decoding_seconds", 0.0) + decode_seconds
        if self.transport.is_encrypted:
            self._timings["he_decryption_seconds"] = self._timings.get("he_decryption_seconds", 0.0) + decode_seconds
        return ServerFeaturePayload(
            client_id=client_id,
            global_node_ids=node_ids.clone(),
            aggregated_features=features,
            num_hops=self.num_hops,
            feature_name="PX",
            metadata={
                "raw_foreign_features_exposed": False,
                "raw_foreign_labels_exposed": False,
                "contains_labels": False,
                "h2_not_cumulative_px_p2x": self.num_hops == 2,
                "feature_pretrain_transport": self.transport.name,
                "feature_pretrain_payload_protection": self.payload_protection.name,
                "recipient_extra_rows_decrypted": 0,
            },
        )

    def install_payload(self, client, payload) -> None:
        if isinstance(payload, list):
            payload = self._decode_blocks_to_payload(client.client_id, payload)
        client.install_feature_payload(payload)

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        communication = None if self._stats is None else self._stats.__dict__
        if communication is not None:
            communication["metadata"] = dict(communication.get("metadata") or {})
            communication["metadata"]["timing"] = dict(self._timings)
        base.update(
            {
                "communication": communication,
                "feature_operator": "P X",
                "normalization": "gcn_norm",
                "self_loops": self.include_self,
                "directionality": self.flow,
                "edge_weight_mode": self.edge_weight_mode,
                "communication_only": True,
                "central_px_shortcut": False,
                "contribution_locality": "owned_features_only",
                "client_receives_full_norm_operator_for_contribution": False,
                "client_visible_propagation_weights": True,
                "client_visible_cross_client_norm_weights": True,
                "topology_private_against_clients": False,
                "client_norm_operator_filter": "source_owned_edges_to_needed_target_rows",
                "raw_foreign_labels_exposed": False,
                "feature_pretrain_transport": self.transport.name,
                "feature_pretrain_payload_protection": self.payload_protection.name,
                "homomorphic_encryption": self.transport.is_encrypted,
                "privacy_enforced": self.transport.is_encrypted
                or bool(self._payload_protection_diagnostics().get("diagnostic_perturbation")),
                "privacy_scope": (
                    "pretraining_transport_and_payload_protection"
                    if self.transport.is_encrypted and self.payload_protection.name != "none"
                    else "pretraining_transport_only"
                    if self.transport.is_encrypted
                    else "pretraining_payload_protection"
                    if self.payload_protection.name != "none"
                    else "none"
                ),
                "secure_aggregation": False,
                "model_updates_encrypted": False,
                "payload_layout": "recipient_aligned_blocks",
                "block_rows": self.block_rows,
                "recipient_row_plans": {
                    client_id: {
                        "rows": int(plan.global_node_ids.numel()),
                        "owned_rows": int(plan.owned_rows.numel()),
                        "placeholder_rows": int(plan.placeholder_rows.numel()),
                        "mode": plan.mode,
                    }
                    for client_id, plan in self._recipient_plans.items()
                },
                "transport": self.transport.diagnostics(),
                "payload_protection": self._payload_protection_diagnostics(),
                "h2_semantics": "PX rows are installed for owned and required one-hop placeholder rows; a linear first layer transforms PX before local normalized aggregation.",
            }
        )
        return base
