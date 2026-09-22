from torch_geometric.nn import GCNConv
import pytest
import torch
from torch_geometric.data import Data

from federated_gcn_aml.experiments.run_federated_amlsim import select_model_class
from federated_gcn_aml.models.gcn import GCN, PXLocalGCN, PreAggregatedGCN


def test_model_selection_by_mode():
    assert select_model_class("cut_edges", 0) is GCN
    assert select_model_class("fedgcn_feature_pretrain", 1) is PreAggregatedGCN
    assert select_model_class("fedgcn_feature_pretrain", 2) is PreAggregatedGCN


def test_pretrain_model_first_layers():
    h1 = PreAggregatedGCN(2, 4, num_layers=2)
    h2 = PreAggregatedGCN(2, 4, num_layers=2)
    legacy_h2 = PXLocalGCN(2, 4, num_layers=2)
    assert h1.convs[0].__class__.__name__ == "Linear"
    assert h2.convs[0].__class__.__name__ == "Linear"
    assert isinstance(legacy_h2.convs[0], GCNConv)


def test_preaggregated_gcn_requires_edge_weight_when_marked_prenormalized():
    model = PreAggregatedGCN(2, 4, num_layers=2)
    data = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    data.edge_weight_is_gcn_norm = True

    with pytest.raises(ValueError, match="expected data.edge_weight"):
        model(data)
