from __future__ import annotations

from typing import Type

import torch

from federated_gcn_aml.federated.payload_protection import PayloadProtectionPolicy
from federated_gcn_aml.federated.messages import (
    BlockPlan,
    ClientFeatureContribution,
    ClientModelUpdateMessage,
    ClientPredictionPacket,
    ClientUpdate,
    FeatureContributionBlock,
    ModelUpdateAggregationPlan,
    PredictionSplitPacket,
    ServerFeaturePayload,
)
from federated_gcn_aml.federated.strategies import (
    build_masked_update_vector,
    extract_non_floating_state,
    flatten_weighted_delta,
)
from federated_gcn_aml.models.gcn import GCN
from federated_gcn_aml.training.trainer import make_pos_weight


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        data,
        model_config: dict,
        lr: float,
        weight_decay: float,
        device: str,
        model_cls: Type[torch.nn.Module] = GCN,
        masked_fedavg_seed: int | None = None,
    ) -> None:
        self.client_id = client_id
        self.data = data.to(device)
        self.model_config = dict(model_config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.model_cls = model_cls
        self.masked_fedavg_seed = masked_fedavg_seed
        self.model = self._new_model()
        self.optimizer_state_policy = "reset_per_round"

    def _new_model(self) -> torch.nn.Module:
        return self.model_cls(**self.model_config).to(self.device)

    def load_global_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({key: value.to(self.device) for key, value in state_dict.items()})

    def install_feature_payload(self, payload: ServerFeaturePayload) -> None:
        expected = self.data.global_node_indices.detach().cpu().long()
        received = payload.global_node_ids.detach().cpu().long()
        if not torch.equal(expected, received):
            raise ValueError(f"Feature payload row order mismatch for client {self.client_id}.")
        self.data.x = payload.aggregated_features.to(self.device).clone()
        self.data.pretrain_payload_metadata = dict(payload.metadata)
        self.data.pretrain_payload_feature_name = payload.feature_name

    def build_feature_contribution(
        self,
        norm_edge_index: torch.Tensor,
        norm_edge_weight: torch.Tensor,
    ) -> ClientFeatureContribution:
        """Build this client's partial normalized feature-aggregation contribution.

        For normalized propagation entries src -> dst with weights w, this client
        contributes along pre-filtered edges whose source node is owned by this
        client:

            C_c[t] = sum_{e: dst_e=t, src_e in V_c} w_e * X[src_e]

        Output is compact by target: target_global_node_ids[j] identifies the
        global target node represented by contribution[j]. The protocol must
        filter normalized edges before calling this method so clients do not
        receive non-owned source entries.
        """

        owned_global = self.data.owned_global_node_indices.detach().cpu().long()
        raw_owned_x = self.data.raw_owned_x.detach().cpu()
        src, dst = norm_edge_index.cpu().long()

        if not src.numel():
            return ClientFeatureContribution(
                client_id=self.client_id,
                target_global_node_ids=torch.empty(0, dtype=torch.long),
                contribution=torch.empty((0, raw_owned_x.size(1)), dtype=raw_owned_x.dtype),
                feature_shape=(0, int(raw_owned_x.size(1))),
                metadata={
                    "source": "owned_raw_features_projected_through_normalized_global_edges",
                    "edges_scanned": int(src.numel()),
                    "edges_selected": 0,
                    "dense_p_materialized": False,
                },
            )

        max_id = owned_global.max().item()
        max_id = max(max_id, src.max().item())
        lookup_size = max_id + 1

        # Dense global-id -> local-owned-feature-row lookup; -1 means not owned.
        source_lookup = torch.full((lookup_size,), -1, dtype=torch.long)
        source_lookup[owned_global] = torch.arange(owned_global.numel(), dtype=torch.long)
        source_rows = source_lookup[src]
        if bool((source_rows < 0).any()):
            raise RuntimeError(
                f"Client {self.client_id} received non-owned source edges for feature contribution construction. "
                "Normalized edges must be filtered by _filtered_norm_edges_for_client before calling "
                "build_feature_contribution."
            )

        selected_weight = norm_edge_weight.cpu().to(raw_owned_x.dtype)
        # Compact target indexing; inverse maps each selected edge to an output row.
        target_ids, inverse = dst.unique(sorted=True, return_inverse=True)
        contribution = torch.zeros((target_ids.numel(), raw_owned_x.size(1)), dtype=raw_owned_x.dtype)
        # Sum duplicate target contributions: contribution[inverse[e]] += weighted_features[e].
        contribution.index_add_(0, inverse, raw_owned_x[source_rows] * selected_weight.view(-1, 1))
        return ClientFeatureContribution(
            client_id=self.client_id,
            target_global_node_ids=target_ids,
            contribution=contribution,
            feature_shape=(int(target_ids.numel()), int(raw_owned_x.size(1))),
            metadata={
                "source": "owned_raw_features_projected_through_normalized_global_edges",
                "edges_scanned": int(src.numel()),
                "edges_selected": int(src.numel()),
                "dense_p_materialized": False,
            },
        )

    def build_feature_contribution_blocks(
        self,
        block_plans: list[BlockPlan],
        norm_edge_index: torch.Tensor,
        norm_edge_weight: torch.Tensor,
        payload_protection: PayloadProtectionPolicy | None = None,
    ) -> list[FeatureContributionBlock]:
        """Frame this client's compact owned-source contribution into blocks.

        The server/protocol supplies the relevant recipient block plans and a
        source-owned filtered normalized operator. The client computes its sparse
        contribution once, then packages matching rows into contribution blocks.
        """

        full_contribution = self.build_feature_contribution(norm_edge_index, norm_edge_weight)
        if payload_protection is not None:
            full_contribution = payload_protection.protect_compact_contribution_before_blocking(
                full_contribution,
                metadata={"client_id": self.client_id},
            )
        target_ids = full_contribution.target_global_node_ids.detach().cpu().long()
        contribution = full_contribution.contribution.detach().cpu()
        if not target_ids.numel() or not block_plans:
            return []

        lookup_size = int(target_ids.max().item()) + 1
        for block_plan in block_plans:
            if block_plan.row_ids.numel():
                lookup_size = max(lookup_size, int(block_plan.row_ids.max().item()) + 1)
        target_lookup = torch.full((lookup_size,), -1, dtype=torch.long)
        target_lookup[target_ids] = torch.arange(target_ids.numel(), dtype=torch.long)

        blocks: list[FeatureContributionBlock] = []
        client_visible_edges = int(norm_edge_index.size(1))
        for block_plan in block_plans:
            block_rows = target_lookup[block_plan.row_ids.long()]
            present = block_rows >= 0
            if not bool(present.any()):
                continue
            block_tensor = torch.zeros(block_plan.shape, dtype=contribution.dtype)
            block_tensor[present] = contribution[block_rows[present]]
            if not bool(torch.any(block_tensor != 0)):
                continue
            if payload_protection is not None:
                payload_protection.record_block_release(self.client_id, block_plan.row_ids[present])
            blocks.append(
                FeatureContributionBlock(
                    source_client_id=self.client_id,
                    recipient_client_id=block_plan.recipient_client_id,
                    block_id=block_plan.block_id,
                    row_ids=block_plan.row_ids.clone(),
                    contribution=block_tensor,
                    feature_shape=block_plan.shape,
                    metadata={
                        "source": "owned_raw_features_projected_to_recipient_block",
                        "edges_scanned": client_visible_edges,
                        "edges_selected": int(present.sum().item()),
                        "dense_p_materialized": False,
                        "client_receives_full_norm_operator_for_contribution": False,
                        "client_norm_operator_filter": "source_owned_edges_to_needed_target_rows",
                    },
                )
            )
        return blocks

    def build_feature_contribution_block(
        self,
        block_plan: BlockPlan,
        norm_edge_index: torch.Tensor,
        norm_edge_weight: torch.Tensor,
    ) -> FeatureContributionBlock:
        """Build one sparse recipient-aligned contribution block.

        Computes P[B, V_k] X[V_k] by filtering normalized edges, avoiding dense
        propagation submatrix materialization.
        """

        owned_global = self.data.owned_global_node_indices.detach().cpu().long()
        raw_owned_x = self.data.raw_owned_x.detach().cpu()
        block_rows = block_plan.row_ids.detach().cpu().long()
        src, dst = norm_edge_index.cpu().long()
        if src.numel():
            max_id = max(int(src.max().item()), int(dst.max().item()), int(owned_global.max().item()), int(block_rows.max().item()))
        else:
            max_id = max(int(owned_global.max().item()), int(block_rows.max().item()))
        lookup_size = max_id + 1

        source_lookup = torch.full((lookup_size,), -1, dtype=torch.long)
        source_lookup[owned_global] = torch.arange(owned_global.numel(), dtype=torch.long)
        target_lookup = torch.full((lookup_size,), -1, dtype=torch.long)
        target_lookup[block_rows] = torch.arange(block_rows.numel(), dtype=torch.long)

        source_rows_all = source_lookup[src]
        target_rows_all = target_lookup[dst]
        selected = (source_rows_all >= 0) & (target_rows_all >= 0)
        contribution = torch.zeros(block_plan.shape, dtype=raw_owned_x.dtype)
        selected_edges = int(selected.sum().item())
        if selected_edges:
            selected_weight = norm_edge_weight.cpu()[selected].to(raw_owned_x.dtype)
            source_rows = source_rows_all[selected]
            target_rows = target_rows_all[selected]
            contribution.index_add_(0, target_rows, raw_owned_x[source_rows] * selected_weight.view(-1, 1))

        return FeatureContributionBlock(
            source_client_id=self.client_id,
            recipient_client_id=block_plan.recipient_client_id,
            block_id=block_plan.block_id,
            row_ids=block_rows.clone(),
            contribution=contribution,
            feature_shape=block_plan.shape,
            metadata={
                "source": "owned_raw_features_projected_to_recipient_block",
                "edges_scanned": int(src.numel()),
                "edges_selected": selected_edges,
                "dense_p_materialized": False,
            },
        )

    def train_local(self, global_state: dict[str, torch.Tensor], local_epochs: int) -> ClientUpdate:
        self.load_global_state(global_state)
        train_count = int(self.data.train_mask.sum().item())
        if train_count == 0:
            return ClientUpdate(
                client_id=self.client_id,
                state_dict={key: value.detach().cpu().clone() for key, value in global_state.items()},
                train_node_count=0,
                loss=None,
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_weight = make_pos_weight(self.data.y, self.data.train_mask)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        last_loss = None
        for _ in range(local_epochs):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(self.data)
            loss = criterion(logits[self.data.train_mask], self.data.y[self.data.train_mask].float())
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        return ClientUpdate(
            client_id=self.client_id,
            state_dict={key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()},
            train_node_count=train_count,
            loss=last_loss,
        )

    def train_node_count(self) -> int:
        return int(self.data.train_mask.sum().item())

    def train_local_model_update(
        self,
        global_state: dict[str, torch.Tensor],
        local_epochs: int,
        aggregation_plan: ModelUpdateAggregationPlan,
    ) -> ClientModelUpdateMessage:
        update = self.train_local(global_state, local_epochs)
        if aggregation_plan.mode == "plain_fedavg":
            return ClientModelUpdateMessage(
                client_id=self.client_id,
                train_node_count=update.train_node_count,
                loss=update.loss,
                state_dict=update.state_dict,
                diagnostics=update.diagnostics,
            )
        if aggregation_plan.mode != "masked_fedavg":
            raise ValueError(f"Unsupported model update aggregation mode: {aggregation_plan.mode}")
        if update.train_node_count <= 0 or self.client_id not in aggregation_plan.participant_client_ids:
            return ClientModelUpdateMessage(
                client_id=self.client_id,
                train_node_count=update.train_node_count,
                loss=None,
                diagnostics={
                    **update.diagnostics,
                    "model_update_aggregation": "masked_fedavg",
                    "masked_update_sent": False,
                },
            )
        if self.masked_fedavg_seed is None:
            raise RuntimeError(
                f"Client {self.client_id} cannot build a masked_fedavg update because no masked FedAvg seed "
                "was configured on the client."
            )

        weight = aggregation_plan.weights[self.client_id]
        weighted_delta = flatten_weighted_delta(
            update.state_dict,
            global_state,
            aggregation_plan.tensor_specs,
            weight,
        )
        masked_update = build_masked_update_vector(
            client_id=self.client_id,
            weighted_delta=weighted_delta,
            plan=aggregation_plan,
            masked_fedavg_seed=int(self.masked_fedavg_seed),
        )
        return ClientModelUpdateMessage(
            client_id=self.client_id,
            train_node_count=update.train_node_count,
            loss=None,
            masked_update_vector=masked_update,
            non_floating_state_dict=extract_non_floating_state(update.state_dict, aggregation_plan.tensor_specs),
            diagnostics={
                **update.diagnostics,
                "model_update_aggregation": "masked_fedavg",
                "masked_update_sent": True,
                "masked_update_vector_numel": int(masked_update.numel()),
            },
        )

    @torch.no_grad()
    def predict_owned(self, global_state: dict[str, torch.Tensor]) -> ClientPredictionPacket:
        self.load_global_state(global_state)
        self.model.eval()
        logits = self.model(self.data).detach().cpu()
        global_node_indices = self.data.global_node_indices.detach().cpu()
        y = self.data.y.detach().cpu()
        val_mask = self.data.val_mask.detach().cpu()
        test_mask = self.data.test_mask.detach().cpu()
        return ClientPredictionPacket(
            client_id=self.client_id,
            val=PredictionSplitPacket("val", global_node_indices[val_mask], y[val_mask], logits[val_mask]),
            test=PredictionSplitPacket("test", global_node_indices[test_mask], y[test_mask], logits[test_mask]),
        )

    def diagnostics(self) -> dict:
        return {
            "client_id": self.client_id,
            "train_nodes": int(self.data.train_mask.sum().item()),
            "val_nodes": int(self.data.val_mask.sum().item()),
            "test_nodes": int(self.data.test_mask.sum().item()),
            "compute_rows": int(self.data.num_nodes),
            "placeholder_context_rows": int(getattr(self.data, "placeholder_context_mask").sum().item()),
            "optimizer_state_policy": self.optimizer_state_policy,
        }
