import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.3):
        super().__init__()

        assert num_layers >= 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, 1)) # binary logits
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)    # apply GCNCov layer
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)    # output logit per node (no activation)

        return x.view(-1) # [num_nodes]


class PreAggregatedGCN(torch.nn.Module):
    """GCN variant for pre-aggregated FedGCN/FedGraph-style input features.

    The pretraining communication step computes Z = P X before local
    optimization. The first trainable layer therefore applies only W_0 to Z:

        H_1 = relu(Z W_0)

    Later layers use local graph convolution. If `data.edge_weight_is_gcn_norm`
    is true, the local graph already stores sliced GCN-normalized propagation
    weights and the layer uses them directly. Otherwise, this class applies PyG
    `gcn_norm` locally to preserve standard `GCNConv` behavior.
    """

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.3):
        super().__init__()

        assert num_layers >= 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))

        self.convs.append(GCNConv(hidden_channels, 1, normalize=False))
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        edge_weight_is_gcn_norm = bool(getattr(data, "edge_weight_is_gcn_norm", False))
        if edge_weight_is_gcn_norm and edge_weight is None:
            raise ValueError("PreAggregatedGCN expected data.edge_weight when edge_weight_is_gcn_norm is true.")
        if not edge_weight_is_gcn_norm:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight=edge_weight,
                num_nodes=data.num_nodes,
                improved=False,
                add_self_loops=True,
                flow="source_to_target",
                dtype=x.dtype,
            )
        x = self.convs[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs[1:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x.view(-1)


class PXLocalGCN(torch.nn.Module):
    """Legacy ablation that applies local GCN propagation over installed PX.

    This class is kept for comparison only. It is not the default faithful h2
    model because its first trainable layer is a GCNConv over already aggregated
    PX rows, whereas the FedGraph/AggreGCN-style path uses a linear first layer.
    """

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.3):
        super().__init__()

        assert num_layers >= 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x.view(-1)


# Deprecated compatibility aliases. Prefer the architecture/protocol names above.
AggregatedFeatureGCN = PreAggregatedGCN
FedGCNAggregatedFeatureGCN = PreAggregatedGCN
FedGCNTwoHopPretrainGCN = PXLocalGCN
