import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from federated_gcn_aml.federated.graph_builder import build_client_graphs
from federated_gcn_aml.federated.partition import build_bank_partition
from federated_gcn_aml.experiments.run_federated_amlsim import parse_args


def _global_graph():
    data = Data(
        x=torch.arange(12, dtype=torch.float).view(4, 3),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
    )
    data.train_mask = torch.tensor([True, True, False, False])
    data.val_mask = torch.tensor([False, False, True, False])
    data.test_mask = torch.tensor([False, False, False, True])
    return data


def _partition():
    accounts = pd.DataFrame({"acct_id": ["a", "b", "c", "d"], "bank_id": ["A", "A", "B", "B"]})
    return build_bank_partition(accounts, ("a", "b", "c", "d"), {"a": 0, "b": 1, "c": 2, "d": 3})


def _edges_as_global(graph):
    gids = graph.global_node_indices
    return sorted((int(gids[s]), int(gids[d])) for s, d in graph.data.edge_index.t().tolist())


def test_cut_edges_keeps_only_intra_bank_edges_and_owned_masks():
    graphs = build_client_graphs(_global_graph(), _partition(), "cut_edges", num_hops=0)
    by_id = {graph.client_id: graph for graph in graphs}
    assert _edges_as_global(by_id["A"]) == [(0, 1), (1, 0)]
    assert _edges_as_global(by_id["B"]) == [(2, 3), (3, 2)]
    assert by_id["A"].diagnostics.raw_foreign_features_exposed is False
    assert by_id["A"].data.owned_mask.all()
    assert by_id["A"].global_node_indices.tolist() == [0, 1]


def test_h2_uses_placeholder_rows_without_raw_foreign_features():
    graphs = build_client_graphs(_global_graph(), _partition(), "fedgcn_feature_pretrain", num_hops=2)
    by_id = {graph.client_id: graph for graph in graphs}
    assert by_id["B"].global_node_indices.tolist() == [0, 2, 3]
    assert _edges_as_global(by_id["B"]) == [(0, 2), (2, 2), (2, 3), (3, 2), (3, 3)]
    assert by_id["B"].diagnostics.placeholder_context_rows == 1
    assert by_id["B"].diagnostics.raw_foreign_features_exposed is False
    context_local = (by_id["B"].global_node_indices == 0).nonzero(as_tuple=False).view(-1)
    assert torch.equal(by_id["B"].data.x[context_local], torch.zeros(1, 3))


def test_no_lhop_subgraph_public_class_mode():
    with pytest.raises(SystemExit):
        parse_args(["--federated-mode", "lhop_subgraph"])
    with pytest.raises(ValueError, match="Unsupported federated mode"):
        build_client_graphs(_global_graph(), _partition(), "lhop_subgraph")  # type: ignore[arg-type]
