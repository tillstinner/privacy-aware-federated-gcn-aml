from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from federated_gcn_aml.federated.messages import ClientGraphDiagnostics
from federated_gcn_aml.federated.partition import BankPartition


FederatedMode = Literal["cut_edges", "fedgcn_feature_pretrain"]
EdgeWeightMode = Literal["unit", "original"]


@dataclass(frozen=True)
class ClientGraph:
    client_id: str
    data: Data
    global_node_indices: torch.Tensor
    owned_global_node_indices: torch.Tensor
    diagnostics: ClientGraphDiagnostics


def inbound_one_hop_nodes(edge_index: torch.Tensor, seed_nodes: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index.cpu()
    seed_mask = torch.zeros(num_nodes, dtype=torch.bool)
    seed_mask[seed_nodes.cpu()] = True
    included = seed_mask.clone()
    included[src[seed_mask[dst]]] = True
    return included.nonzero(as_tuple=False).view(-1).long()


def _edge_weight_for_mask(global_data: Data, edge_mask: torch.Tensor, mode: EdgeWeightMode) -> torch.Tensor:
    if mode == "unit":
        return torch.ones(int(edge_mask.sum().item()), dtype=torch.float)
    if mode == "original":
        edge_weight = getattr(global_data, "edge_weight", None)
        if edge_weight is None:
            return torch.ones(int(edge_mask.sum().item()), dtype=torch.float)
        return edge_weight.cpu()[edge_mask].clone().float()
    raise ValueError(f"Unsupported edge_weight_mode: {mode}")


def _make_client_graph(
    client_id: str,
    global_data: Data,
    owned_global_nodes: torch.Tensor,
    compute_global_nodes: torch.Tensor,
    mode: FederatedMode,
    num_hops: int,
    edge_weight_mode: EdgeWeightMode,
) -> ClientGraph:
    num_global_nodes = int(global_data.num_nodes)
    owned_global_nodes = owned_global_nodes.unique(sorted=True).cpu()
    compute_global_nodes = compute_global_nodes.unique(sorted=True).cpu()

    node_mask = torch.zeros(num_global_nodes, dtype=torch.bool)
    node_mask[compute_global_nodes] = True
    owned_global_mask = torch.zeros(num_global_nodes, dtype=torch.bool)
    owned_global_mask[owned_global_nodes] = True

    local_index = torch.full((num_global_nodes,), -1, dtype=torch.long)
    local_index[compute_global_nodes] = torch.arange(compute_global_nodes.numel(), dtype=torch.long)

    if mode == "fedgcn_feature_pretrain" and num_hops == 2:
        base_edge_weight = None
        if edge_weight_mode == "original":
            base_edge_weight = getattr(global_data, "edge_weight", None)
        norm_edge_index, norm_edge_weight = gcn_norm(
            global_data.edge_index.cpu(),
            edge_weight=None if base_edge_weight is None else base_edge_weight.cpu().float(),
            num_nodes=num_global_nodes,
            improved=False,
            add_self_loops=True,
            flow="source_to_target",
            dtype=global_data.x.dtype,
        )
        src, dst = norm_edge_index.cpu()
        edge_mask = node_mask[src] & owned_global_mask[dst]
        local_edge_weight = norm_edge_weight.cpu()[edge_mask].clone().float()
    else:
        src, dst = global_data.edge_index.cpu()
        edge_mask = owned_global_mask[src] & owned_global_mask[dst]
        local_edge_weight = _edge_weight_for_mask(global_data, edge_mask, edge_weight_mode)
    local_edge_index = torch.stack([local_index[src[edge_mask]], local_index[dst[edge_mask]]])

    x = torch.zeros((compute_global_nodes.numel(), global_data.x.size(1)), dtype=global_data.x.dtype)
    y = torch.zeros(compute_global_nodes.numel(), dtype=global_data.y.dtype)
    owned_local_nodes = local_index[owned_global_nodes]
    if bool((owned_local_nodes < 0).any()):
        raise ValueError(f"Owned nodes for client {client_id} are missing from the compute graph.")
    x[owned_local_nodes] = global_data.x.cpu()[owned_global_nodes]
    y[owned_local_nodes] = global_data.y.cpu()[owned_global_nodes]

    data = Data(x=x.clone(), edge_index=local_edge_index, y=y.clone())
    data.edge_weight = local_edge_weight
    data.edge_weight_is_gcn_norm = mode == "fedgcn_feature_pretrain" and num_hops == 2

    owned_local_mask = torch.zeros(compute_global_nodes.numel(), dtype=torch.bool)
    owned_local_mask[owned_local_nodes] = True
    data.owned_mask = owned_local_mask
    data.placeholder_context_mask = ~owned_local_mask
    data.train_mask = owned_local_mask & global_data.train_mask.cpu()[compute_global_nodes]
    data.val_mask = owned_local_mask & global_data.val_mask.cpu()[compute_global_nodes]
    data.test_mask = owned_local_mask & global_data.test_mask.cpu()[compute_global_nodes]

    local_src_global = src[edge_mask]
    local_dst_global = dst[edge_mask]
    internal_edge_mask = owned_global_mask[local_src_global] & owned_global_mask[local_dst_global]
    cross_edge_references = int((edge_mask & ~(owned_global_mask[src] & owned_global_mask[dst])).sum().item())
    payload_rows = int(compute_global_nodes.numel()) if mode == "fedgcn_feature_pretrain" else 0
    diagnostics = ClientGraphDiagnostics(
        client_id=client_id,
        owned_nodes=int(owned_global_nodes.numel()),
        compute_rows=int(compute_global_nodes.numel()),
        placeholder_context_rows=int((~owned_local_mask).sum().item()),
        raw_foreign_features_exposed=False,
        edges=int(local_edge_index.size(1)),
        owned_internal_edges=int(internal_edge_mask.sum().item()),
        cross_edge_references=cross_edge_references,
        train_nodes=int(data.train_mask.sum().item()),
        val_nodes=int(data.val_mask.sum().item()),
        test_nodes=int(data.test_mask.sum().item()),
        positive_train_labels=int(data.y[data.train_mask].sum().item()),
        positive_val_labels=int(data.y[data.val_mask].sum().item()),
        positive_test_labels=int(data.y[data.test_mask].sum().item()),
        pretrain_payload_rows=payload_rows,
        pretrain_feature_source="server_payload_px" if mode == "fedgcn_feature_pretrain" else "none",
        communication_only=mode == "fedgcn_feature_pretrain",
    )

    data.global_node_indices = compute_global_nodes
    data.owned_global_node_indices = owned_global_nodes
    data.raw_owned_x = global_data.x.cpu()[owned_global_nodes].clone()
    return ClientGraph(
        client_id=client_id,
        data=data,
        global_node_indices=compute_global_nodes,
        owned_global_node_indices=owned_global_nodes,
        diagnostics=diagnostics,
    )


def build_client_graphs(
    global_data: Data,
    partition: BankPartition,
    mode: FederatedMode,
    num_hops: int = 0,
    edge_weight_mode: EdgeWeightMode = "unit",
) -> list[ClientGraph]:
    if mode not in {"cut_edges", "fedgcn_feature_pretrain"}:
        raise ValueError(f"Unsupported federated mode: {mode}")
    if edge_weight_mode not in {"unit", "original"}:
        raise ValueError(f"Unsupported edge_weight_mode: {edge_weight_mode}")
    if mode == "cut_edges":
        num_hops = 0
    elif num_hops not in {1, 2}:
        raise ValueError("fedgcn_feature_pretrain requires num_hops to be 1 or 2")

    client_graphs = []
    for client_id in partition.bank_ids:
        owned_nodes = partition.client_node_indices[client_id]
        if mode == "fedgcn_feature_pretrain" and num_hops == 2:
            compute_nodes = inbound_one_hop_nodes(global_data.edge_index, owned_nodes, global_data.num_nodes)
        else:
            compute_nodes = owned_nodes
        client_graphs.append(
            _make_client_graph(
                client_id=client_id,
                global_data=global_data,
                owned_global_nodes=owned_nodes,
                compute_global_nodes=compute_nodes,
                mode=mode,
                num_hops=num_hops,
                edge_weight_mode=edge_weight_mode,
            )
        )
    return client_graphs
