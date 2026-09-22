import pandas as pd
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from federated_gcn_aml.federated.client import FederatedClient
from federated_gcn_aml.federated.graph_builder import build_client_graphs
from federated_gcn_aml.federated.messages import BlockPlan, ClientFeatureContribution
from federated_gcn_aml.federated.partition import build_bank_partition
from federated_gcn_aml.federated.payload_protection import (
    CentralDiagnosticPerturbation,
    CentralRowDPPayloadProtection,
    LocalDiagnosticPerturbation,
    LocalRowDPPayloadProtection,
    analytic_gaussian_sigma,
    classical_gaussian_sigma,
)
from federated_gcn_aml.federated.protocols import FedGCNFeaturePretrainProtocol
from federated_gcn_aml.federated.strategies import FedAvgStrategy
from federated_gcn_aml.federated.transports import PlainTensorTransport, TenSEALCKKSTransport
from federated_gcn_aml.experiments.run_federated_amlsim import parse_args
from federated_gcn_aml.models.gcn import GCN, PXLocalGCN, PreAggregatedGCN


def _data():
    data = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
    )
    data.train_mask = torch.tensor([True, True, False, False])
    data.val_mask = torch.tensor([False, False, True, False])
    data.test_mask = torch.tensor([False, False, False, True])
    return data


def _partition():
    accounts = pd.DataFrame({"acct_id": ["a", "b", "c", "d"], "bank_id": ["A", "A", "B", "B"]})
    return build_bank_partition(accounts, ("a", "b", "c", "d"), {"a": 0, "b": 1, "c": 2, "d": 3})


def _expected_px(data):
    edge_index, edge_weight = gcn_norm(data.edge_index, None, data.num_nodes, add_self_loops=True, flow="source_to_target", dtype=data.x.dtype)
    out = torch.zeros_like(data.x)
    src, dst = edge_index
    out.index_add_(0, dst, data.x[src] * edge_weight.view(-1, 1))
    return out


def _owned_source_contribution_oracle(data, owned_global, edge_index, edge_weight):
    out = torch.zeros_like(data.x)
    owned = set(int(node) for node in owned_global.tolist())
    src, dst = edge_index.cpu().long()
    for edge_idx, (source, target) in enumerate(zip(src.tolist(), dst.tolist())):
        if int(source) in owned:
            out[int(target)] += data.x[int(source)] * edge_weight[edge_idx]
    return out


def _clip_rows(tensor, clip_norm):
    norms = tensor.norm(p=2, dim=1)
    scale = torch.ones_like(norms)
    clipped = norms > clip_norm
    scale[clipped] = clip_norm / norms[clipped].clamp_min(1e-12)
    return tensor * scale.view(-1, 1)


def _expected_clipped_source_px(data, partition, graphs, clients, protocol, clip_norm):
    expected = torch.zeros_like(data.x)
    for client in clients:
        edge_index, edge_weight = protocol._filtered_norm_edges_for_client(client)
        contribution = client.build_feature_contribution(edge_index, edge_weight)
        clipped = _clip_rows(contribution.contribution, clip_norm)
        expected[contribution.target_global_node_ids.long()] += clipped.to(expected.dtype)
    return expected


def test_h1_protocol_installs_owned_px_only():
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=1)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(num_hops=1, feature_pretrain_transport="plain")
    protocol.prepare(data, partition, graphs)
    protocol.aggregate_on_server(protocol.collect_contributions(clients))
    communication = protocol.diagnostics()["communication"]["metadata"]
    assert communication["uploaded_row_occurrences"] == 6
    assert communication["unique_uploaded_source_target_rows"] == 6
    assert communication["duplicate_upload_factor"] == 1.0
    assert communication["duplicate_placeholder_rows_h2"] == 0
    for client in clients:
        payload = protocol.build_payload_for_client(client.client_id)
        protocol.install_payload(client, payload)
        assert torch.allclose(client.data.x.cpu(), _expected_px(data)[client.data.global_node_indices.cpu()])
        assert client.data.placeholder_context_mask.sum().item() == 0
        assert payload.metadata["raw_foreign_features_exposed"] is False
        assert payload.metadata["feature_pretrain_transport"] == "plain"


def test_h2_protocol_installs_px_for_placeholders_not_cumulative_px_p2x():
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=2)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(num_hops=2, feature_pretrain_transport="plain", block_rows=2)
    protocol.prepare(data, partition, graphs)
    protocol.aggregate_on_server(protocol.collect_contributions(clients))
    communication = protocol.diagnostics()["communication"]["metadata"]
    assert communication["uploaded_row_occurrences"] == 8
    assert communication["unique_uploaded_source_target_rows"] == 6
    assert communication["duplicate_upload_factor"] > 1.0
    assert communication["duplicate_placeholder_rows_h2"] == 2
    client_b = [client for client in clients if client.client_id == "B"][0]
    payload = protocol.build_payload_for_client("B")
    protocol.install_payload(client_b, payload)
    px = _expected_px(data)
    edge_index, edge_weight = gcn_norm(data.edge_index, None, data.num_nodes, add_self_loops=True, flow="source_to_target", dtype=data.x.dtype)
    p2x = torch.zeros_like(data.x)
    src, dst = edge_index
    p2x.index_add_(0, dst, px[src] * edge_weight.view(-1, 1))
    assert torch.allclose(client_b.data.x.cpu(), px[client_b.data.global_node_indices.cpu()])
    assert not torch.allclose(client_b.data.x.cpu(), (px + p2x)[client_b.data.global_node_indices.cpu()])
    assert client_b.data.placeholder_context_mask.any()
    assert client_b.data.train_mask[client_b.data.placeholder_context_mask].sum().item() == 0


def test_h2_faithful_two_layer_forward_matches_px_linear_then_p_slice():
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=2)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 3, "num_layers": 2, "dropout": 0.0}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(num_hops=2)
    protocol.prepare(data, partition, graphs)
    protocol.aggregate_on_server(protocol.collect_contributions(clients))
    client_b = [client for client in clients if client.client_id == "B"][0]
    protocol.install_payload(client_b, protocol.build_payload_for_client("B"))

    model = client_b.model
    model.eval()
    with torch.no_grad():
        model.convs[0].weight.copy_(torch.tensor([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.7]]))
        model.convs[0].bias.copy_(torch.tensor([0.05, -0.1, 0.2]))
        model.convs[1].lin.weight.copy_(torch.tensor([[0.6, -0.2, 0.3]]))
        model.convs[1].bias.copy_(torch.tensor([0.15]))

    logits = model(client_b.data).detach().cpu()
    hidden = torch.relu(client_b.data.x.cpu() @ model.convs[0].weight.detach().cpu().t() + model.convs[0].bias.detach().cpu())
    transformed = hidden @ model.convs[1].lin.weight.detach().cpu().t()
    expected = torch.zeros(client_b.data.num_nodes, 1)
    src, dst = client_b.data.edge_index.cpu()
    expected.index_add_(0, dst, transformed[src] * client_b.data.edge_weight.cpu().view(-1, 1))
    expected = expected.view(-1) + model.convs[1].bias.detach().cpu()

    owned = client_b.data.owned_mask.cpu()
    assert torch.allclose(logits[owned], expected[owned], atol=1e-6)

    legacy = PXLocalGCN(2, 3, num_layers=2, dropout=0.0)
    legacy.eval()
    with torch.no_grad():
        legacy.convs[0].lin.weight.copy_(model.convs[0].weight)
        legacy.convs[0].bias.copy_(model.convs[0].bias)
        legacy.convs[1].lin.weight.copy_(model.convs[1].lin.weight)
        legacy.convs[1].bias.copy_(model.convs[1].bias)
    legacy_logits = legacy(client_b.data).detach().cpu()
    assert not torch.allclose(legacy_logits[owned], logits[owned])


def test_sparse_block_contribution_matches_dense_oracle():
    data = _data()
    partition = _partition()
    graph_a = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=1)[0]
    client = FederatedClient(
        graph_a.client_id,
        graph_a.data,
        {"in_channels": 2, "hidden_channels": 4, "num_layers": 2},
        0.01,
        0.0,
        "cpu",
        PreAggregatedGCN,
    )
    edge_index, edge_weight = gcn_norm(
        data.edge_index,
        None,
        data.num_nodes,
        add_self_loops=True,
        flow="source_to_target",
        dtype=data.x.dtype,
    )
    block = BlockPlan("B", "B:0", torch.tensor([2, 3]), (2, 2), 0, 2)
    contribution = client.build_feature_contribution_block(block, edge_index, edge_weight)
    dense = torch.zeros_like(data.x)
    owned = set(client.data.owned_global_node_indices.tolist())
    src, dst = edge_index
    for edge_idx, (source, target) in enumerate(zip(src.tolist(), dst.tolist())):
        if source in owned:
            dense[target] += data.x[source] * edge_weight[edge_idx]
    assert contribution.metadata["dense_p_materialized"] is False
    assert torch.allclose(contribution.contribution, dense[block.row_ids])


def test_client_builds_relevant_contribution_blocks_from_block_plan_list():
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=2)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(num_hops=2, feature_pretrain_transport="plain", block_rows=2)
    protocol.prepare(data, partition, graphs)
    client = clients[0]
    edge_index, edge_weight = protocol._filtered_norm_edges_for_client(client)
    target_rows = edge_index[1].unique(sorted=True)
    block_plans = protocol._block_plans_for_target_rows(target_rows)
    blocks = client.build_feature_contribution_blocks(block_plans, edge_index, edge_weight)

    oracle = _owned_source_contribution_oracle(data, client.data.owned_global_node_indices, protocol._norm_edge_index, protocol._norm_edge_weight)
    assert blocks
    for block in blocks:
        assert block.source_client_id == client.client_id
        assert torch.allclose(block.contribution, oracle[block.row_ids])
        assert block.metadata["dense_p_materialized"] is False
        assert block.metadata["client_receives_full_norm_operator_for_contribution"] is False


def test_protocol_filters_norm_edges_before_client_contribution():
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=2)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(num_hops=2, feature_pretrain_transport="plain", block_rows=2)
    protocol.prepare(data, partition, graphs)
    full_edge_count = int(protocol._norm_edge_index.size(1))
    needed_rows = torch.cat([block.row_ids for block in protocol._block_plans]).unique()

    for client in clients:
        edge_index, edge_weight = protocol._filtered_norm_edges_for_client(client)
        assert edge_weight.numel() == edge_index.size(1)
        assert 0 < edge_index.size(1) < full_edge_count
        src, dst = edge_index
        owned = set(client.data.owned_global_node_indices.tolist())
        needed = set(needed_rows.tolist())
        assert all(int(node) in owned for node in src.tolist())
        assert all(int(node) in needed for node in dst.tolist())
        relevant_block_plans = protocol._block_plans_for_target_rows(dst.unique(sorted=True))
        relevant_ids = {(plan.recipient_client_id, plan.block_id) for plan in relevant_block_plans}
        for plan in protocol._block_plans:
            intersects = bool(torch.isin(plan.row_ids, dst.unique(sorted=True)).any())
            assert ((plan.recipient_client_id, plan.block_id) in relevant_ids) == intersects

        filtered_contribution = client.build_feature_contribution(edge_index, edge_weight)
        oracle = _owned_source_contribution_oracle(data, client.data.owned_global_node_indices, protocol._norm_edge_index, protocol._norm_edge_weight)
        filtered_map = {
            int(row): filtered_contribution.contribution[idx]
            for idx, row in enumerate(filtered_contribution.target_global_node_ids.tolist())
        }
        assert set(filtered_map).issubset(needed)
        for row, tensor in filtered_map.items():
            assert torch.allclose(tensor, oracle[row])


def test_build_feature_contribution_rejects_non_owned_source_edges():
    data = _data()
    partition = _partition()
    graph_a = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=1)[0]
    client = FederatedClient(
        graph_a.client_id,
        graph_a.data,
        {"in_channels": 2, "hidden_channels": 4, "num_layers": 2},
        0.01,
        0.0,
        "cpu",
        PreAggregatedGCN,
    )
    edge_index, edge_weight = gcn_norm(
        data.edge_index,
        None,
        data.num_nodes,
        add_self_loops=True,
        flow="source_to_target",
        dtype=data.x.dtype,
    )
    with pytest.raises(RuntimeError, match="non-owned source edges"):
        client.build_feature_contribution(edge_index, edge_weight)


def test_build_feature_contribution_empty_filtered_edges_returns_empty():
    data = _data()
    partition = _partition()
    graph_a = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=1)[0]
    client = FederatedClient(
        graph_a.client_id,
        graph_a.data,
        {"in_channels": 2, "hidden_channels": 4, "num_layers": 2},
        0.01,
        0.0,
        "cpu",
        PreAggregatedGCN,
    )
    contribution = client.build_feature_contribution(
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0,), dtype=data.x.dtype),
    )
    assert contribution.target_global_node_ids.numel() == 0
    assert contribution.contribution.shape == (0, data.x.size(1))
    assert contribution.metadata["edges_scanned"] == 0
    assert contribution.metadata["edges_selected"] == 0


def test_build_feature_contribution_sums_duplicate_targets():
    data = _data()
    partition = _partition()
    graph_a = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=1)[0]
    client = FederatedClient(
        graph_a.client_id,
        graph_a.data,
        {"in_channels": 2, "hidden_channels": 4, "num_layers": 2},
        0.01,
        0.0,
        "cpu",
        PreAggregatedGCN,
    )
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.25], dtype=data.x.dtype)
    contribution = client.build_feature_contribution(edge_index, edge_weight)
    assert contribution.target_global_node_ids.tolist() == [2]
    expected = data.x[0] * 0.5 + data.x[1] * 0.25
    assert torch.allclose(contribution.contribution[0], expected)
    assert contribution.metadata["edges_selected"] == 2


def test_plain_transport_exact_sum():
    transport = PlainTensorTransport()
    block = BlockPlan("A", "A:0", torch.tensor([0, 1]), (2, 2), 0, 2)
    first = transport.encode_contribution_block(
        client_block("A", "A", "A:0", block.row_ids, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    )
    second = transport.encode_contribution_block(
        client_block("B", "A", "A:0", block.row_ids, torch.tensor([[0.5, 1.0], [1.5, 2.0]]))
    )
    payload = transport.aggregate_blocks([first, second], block)
    assert not payload.is_encrypted
    assert torch.allclose(transport.decode_payload_block(payload), torch.tensor([[1.5, 3.0], [4.5, 6.0]]))


def test_diagnostic_clipping_leaves_small_rows_and_caps_large_rows():
    policy = CentralDiagnosticPerturbation(clip_norm=2.0, noise_multiplier=0.0, diagnostic_seed=7)
    contribution = ClientFeatureContribution(
        client_id="A",
        target_global_node_ids=torch.tensor([0, 1]),
        contribution=torch.tensor([[1.0, 1.0], [3.0, 4.0]]),
        feature_shape=(2, 2),
    )
    protected = policy.protect_compact_contribution_before_blocking(contribution)
    assert torch.allclose(protected.contribution[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(protected.contribution[1], torch.tensor([1.2, 1.6]))
    diagnostics = policy.diagnostics()
    assert diagnostics["rows_clipped"] == 1
    assert diagnostics["clip_fraction"] == 0.5


def test_local_diagnostic_seed_reproducibility_and_variation():
    contribution = ClientFeatureContribution(
        client_id="A",
        target_global_node_ids=torch.tensor([0, 1]),
        contribution=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        feature_shape=(2, 2),
    )
    first = LocalDiagnosticPerturbation(clip_norm=10.0, noise_multiplier=0.5, diagnostic_seed=11)
    second = LocalDiagnosticPerturbation(clip_norm=10.0, noise_multiplier=0.5, diagnostic_seed=11)
    third = LocalDiagnosticPerturbation(clip_norm=10.0, noise_multiplier=0.5, diagnostic_seed=12)
    first_out = first.protect_compact_contribution_before_blocking(contribution).contribution
    second_out = second.protect_compact_contribution_before_blocking(contribution).contribution
    third_out = third.protect_compact_contribution_before_blocking(contribution).contribution
    assert torch.allclose(first_out, second_out)
    assert not torch.allclose(first_out, third_out)
    assert first.diagnostics()["rows_noised"] == 2


def test_plain_diagnostic_modes_zero_noise_equal_clipped_oracle():
    data = _data()
    partition = _partition()
    clip_norm = 1.0
    expected = None
    for policy, canonical in (
        ("central_diagnostic_perturbation", "central_diagnostic_perturbation"),
        ("local_diagnostic_perturbation", "local_diagnostic_perturbation"),
    ):
        graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=2)
        clients = [
            FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
            for g in graphs
        ]
        protocol = FedGCNFeaturePretrainProtocol(
            num_hops=2,
            feature_pretrain_transport="plain",
            feature_pretrain_payload_protection=policy,
            payload_clip_norm=clip_norm,
            diagnostic_noise_multiplier=0.0,
            diagnostic_seed=123,
            block_rows=2,
        )
        protocol.prepare(data, partition, graphs)
        if expected is None:
            expected = _expected_clipped_source_px(data, partition, graphs, clients, protocol, clip_norm)
        protocol.aggregate_on_server(protocol.collect_contributions(clients))
        communication = protocol.diagnostics()["communication"]["metadata"]
        assert communication["payload_protection_policy"] == canonical
        assert communication["differential_privacy"] is False
        assert communication["diagnostic_perturbation"] is True
        assert communication["diagnostic_perturbation_mechanism"] == "gaussian"
        assert communication["diagnostic_seed"] == 123
        assert communication["formal_dp_accounting"] is False
        assert communication["payload_protection_kind"] == "diagnostic_gaussian_perturbation"
        assert communication["diagnostic_noise_multiplier"] == 0.0
        assert communication["payload_dp_claim"] is None
        assert communication["payload_dp_epsilon"] is None
        assert communication["rows_clipped"] > 0
        for client in clients:
            payload = protocol.build_payload_for_client(client.client_id)
            protocol.install_payload(client, payload)
            assert torch.allclose(client.data.x.cpu(), expected[client.data.global_node_indices.cpu()])


def test_he_diagnostic_modes_zero_noise_match_plain_clipped_oracle():
    pytest.importorskip("tenseal")
    data = _data()
    partition = _partition()
    clip_norm = 1.0
    for policy in ("central_diagnostic_perturbation", "local_diagnostic_perturbation"):
        graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=1)
        clients = [
            FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
            for g in graphs
        ]
        protocol = FedGCNFeaturePretrainProtocol(
            num_hops=1,
            feature_pretrain_transport="he",
            feature_pretrain_payload_protection=policy,
            payload_clip_norm=clip_norm,
            diagnostic_noise_multiplier=0.0,
            diagnostic_seed=123,
            block_rows=2,
        )
        protocol.prepare(data, partition, graphs)
        expected = _expected_clipped_source_px(data, partition, graphs, clients, protocol, clip_norm)
        protocol.aggregate_on_server(protocol.collect_contributions(clients))
        communication = protocol.diagnostics()["communication"]["metadata"]
        assert communication["ckks_payload_max_abs_error"] <= 1e-3
        assert communication["ckks_payload_mean_abs_error"] <= 1e-4
        for client in clients:
            protocol.install_payload(client, protocol.build_payload_for_client(client.client_id))
            assert torch.allclose(client.data.x.cpu(), expected[client.data.global_node_indices.cpu()], atol=1e-3)


def test_formal_row_dp_classical_gaussian_calibration_metadata():
    policy = LocalRowDPPayloadProtection(clip_norm=2.0, epsilon=1.0, delta=1e-5, payload_dp_seed=17)
    diagnostics = policy.diagnostics()
    assert diagnostics["formal_dp_accounting"] is True
    assert diagnostics["formal_differential_privacy"] is True
    assert diagnostics["payload_dp_claim"] == "contribution_row_substitution_dp"
    assert diagnostics["payload_dp_sensitivity_l2"] == 4.0
    assert diagnostics["payload_dp_noise_multiplier"] is None
    assert diagnostics["payload_dp_noise_std"] == pytest.approx(
        classical_gaussian_sigma(epsilon=1.0, delta=1e-5, sensitivity_l2=4.0)
    )
    assert diagnostics["dp_epsilon"] == 1.0
    assert diagnostics["dp_delta"] == 1e-5
    assert diagnostics["not_claimed"] == [
        "account_level_dp",
        "transaction_level_dp",
        "edge_topology_dp",
        "client_level_dp",
        "model_update_dp",
        "end_to_end_dp",
    ]


def test_analytic_gaussian_calibration_validation_and_invariants():
    with pytest.raises(ValueError, match="epsilon <= 1.0"):
        classical_gaussian_sigma(epsilon=2.0, delta=1e-5, sensitivity_l2=2.0)
    with pytest.raises(ValueError, match="Unsupported payload DP calibration"):
        LocalRowDPPayloadProtection(
            clip_norm=1.0,
            epsilon=1.0,
            delta=1e-5,
            payload_dp_seed=17,
            calibration="bogus_gaussian",
        )
    invalid_cases = [
        {"epsilon": 0.0, "delta": 1e-5, "sensitivity_l2": 2.0},
        {"epsilon": 1.0, "delta": 0.0, "sensitivity_l2": 2.0},
        {"epsilon": 1.0, "delta": 1.0, "sensitivity_l2": 2.0},
        {"epsilon": 1.0, "delta": 1e-5, "sensitivity_l2": 0.0},
    ]
    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            analytic_gaussian_sigma(**kwargs)
    with pytest.raises(ValueError, match="epsilon is too large"):
        analytic_gaussian_sigma(epsilon=1_000.0, delta=1e-5, sensitivity_l2=2.0)

    sigma_eps_1 = analytic_gaussian_sigma(epsilon=1.0, delta=1e-5, sensitivity_l2=2.0)
    sigma_eps_2 = analytic_gaussian_sigma(epsilon=2.0, delta=1e-5, sensitivity_l2=2.0)
    sigma_eps_8 = analytic_gaussian_sigma(epsilon=8.0, delta=1e-5, sensitivity_l2=2.0)
    sigma_delta_1e6 = analytic_gaussian_sigma(epsilon=2.0, delta=1e-6, sensitivity_l2=2.0)
    sigma_sens_4 = analytic_gaussian_sigma(epsilon=2.0, delta=1e-5, sensitivity_l2=4.0)

    assert sigma_eps_1 == pytest.approx(7.461263269631884, rel=1e-10)
    assert sigma_eps_2 == pytest.approx(3.987624891287073, rel=1e-10)
    assert sigma_eps_8 == pytest.approx(1.200458144397903, rel=1e-10)
    assert sigma_eps_8 > 0.0
    assert sigma_eps_8 < sigma_eps_2 < sigma_eps_1
    assert sigma_delta_1e6 > sigma_eps_2
    assert sigma_sens_4 == pytest.approx(2.0 * sigma_eps_2, rel=1e-10)


def test_formal_row_dp_analytic_gaussian_calibration_metadata():
    policy = CentralRowDPPayloadProtection(
        clip_norm=1.0,
        epsilon=2.0,
        delta=1e-5,
        payload_dp_seed=17,
        calibration="analytic_gaussian",
    )
    diagnostics = policy.diagnostics()
    assert diagnostics["formal_dp_accounting"] is True
    assert diagnostics["formal_differential_privacy"] is True
    assert diagnostics["payload_dp_calibration"] == "analytic_gaussian"
    assert diagnostics["payload_dp_sensitivity_l2"] == 2.0
    assert diagnostics["payload_dp_noise_multiplier"] is None
    assert diagnostics["payload_dp_noise_std"] == pytest.approx(
        analytic_gaussian_sigma(epsilon=2.0, delta=1e-5, sensitivity_l2=2.0),
        rel=1e-10,
    )
    assert diagnostics["payload_dp_epsilon"] == 2.0
    assert diagnostics["payload_dp_delta"] == 1e-5


def test_local_row_dp_reuses_noised_compact_rows_for_duplicate_h2_blocks():
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=2)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(
        num_hops=2,
        feature_pretrain_transport="plain",
        feature_pretrain_payload_protection="local_row_dp",
        payload_clip_norm=10.0,
        payload_dp_epsilon=1.0,
        payload_dp_delta=1e-5,
        payload_dp_seed=123,
        block_rows=2,
    )
    protocol.prepare(data, partition, graphs)
    protocol.aggregate_on_server(protocol.collect_contributions(clients))
    communication = protocol.diagnostics()["communication"]["metadata"]
    assert communication["formal_dp_accounting"] is True
    assert communication["duplicate_release_policy"] == "reuse_noised_value"
    assert communication["max_release_count_per_unit"] > 1

    payload_a = protocol.build_payload_for_client("A")
    payload_b = protocol.build_payload_for_client("B")
    rows_a = payload_a.global_node_ids.tolist()
    rows_b = payload_b.global_node_ids.tolist()
    common_rows = sorted(set(rows_a) & set(rows_b))
    assert common_rows
    for row in common_rows:
        value_a = payload_a.aggregated_features[rows_a.index(row)]
        value_b = payload_b.aggregated_features[rows_b.index(row)]
        assert torch.allclose(value_a, value_b)


def test_central_row_dp_reuses_noise_for_duplicate_h2_aggregate_rows():
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=2)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(
        num_hops=2,
        feature_pretrain_transport="plain",
        feature_pretrain_payload_protection="central_row_dp",
        payload_clip_norm=10.0,
        payload_dp_epsilon=2.0,
        payload_dp_delta=1e-5,
        payload_dp_calibration="analytic_gaussian",
        payload_dp_seed=123,
        block_rows=2,
    )
    protocol.prepare(data, partition, graphs)
    expected_private_units = set()
    for client in clients:
        edge_index, edge_weight = protocol._filtered_norm_edges_for_client(client)
        contribution = client.build_feature_contribution(edge_index, edge_weight)
        for row in contribution.target_global_node_ids.tolist():
            expected_private_units.add((client.client_id, int(row)))
    protocol.aggregate_on_server(protocol.collect_contributions(clients))
    communication = protocol.diagnostics()["communication"]["metadata"]
    assert communication["formal_dp_accounting"] is True
    assert communication["payload_dp_calibration"] == "analytic_gaussian"
    assert communication["payload_dp_epsilon"] == 2.0
    assert communication["payload_dp_noise_std"] == pytest.approx(
        analytic_gaussian_sigma(epsilon=2.0, delta=1e-5, sensitivity_l2=20.0),
        rel=1e-10,
    )
    assert communication["duplicate_release_policy"] == "reuse_noised_value"
    assert communication["num_noised_releases"] < communication["num_unique_private_units"]
    assert communication["num_unique_private_units"] == len(expected_private_units)
    assert communication["max_release_count_per_unit"] > 1

    payload_a = protocol.build_payload_for_client("A")
    payload_b = protocol.build_payload_for_client("B")
    rows_a = payload_a.global_node_ids.tolist()
    rows_b = payload_b.global_node_ids.tolist()
    common_rows = sorted(set(rows_a) & set(rows_b))
    assert common_rows
    for row in common_rows:
        value_a = payload_a.aggregated_features[rows_a.index(row)]
        value_b = payload_b.aggregated_features[rows_b.index(row)]
        assert torch.allclose(value_a, value_b)


def test_he_formal_row_dp_disables_clean_payload_oracle_diagnostics():
    pytest.importorskip("tenseal")
    data = _data()
    partition = _partition()
    graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=1)
    clients = [
        FederatedClient(g.client_id, g.data, {"in_channels": 2, "hidden_channels": 4, "num_layers": 2}, 0.01, 0.0, "cpu", PreAggregatedGCN)
        for g in graphs
    ]
    protocol = FedGCNFeaturePretrainProtocol(
        num_hops=1,
        feature_pretrain_transport="he",
        feature_pretrain_payload_protection="central_row_dp",
        payload_clip_norm=10.0,
        payload_dp_epsilon=1.0,
        payload_dp_delta=1e-5,
        payload_dp_seed=123,
        block_rows=2,
    )
    protocol.prepare(data, partition, graphs)
    protocol.aggregate_on_server(protocol.collect_contributions(clients))
    communication = protocol.diagnostics()["communication"]["metadata"]
    assert communication["formal_dp_oracle_diagnostics_enabled"] is False
    assert communication["diagnostics_scope"] == "no_clean_payload_oracle_for_formal_dp"
    assert communication["ckks_payload_error_sampled"] is None
    assert communication["ckks_payload_max_abs_error"] is None
    assert communication["ckks_payload_mean_abs_error"] is None
    assert communication["trust_boundary"] == "central_he_public_context"
    for client in clients:
        protocol.install_payload(client, protocol.build_payload_for_client(client.client_id))
        assert client.data.x.shape[1] == data.x.shape[1]


def client_block(source, recipient, block_id, rows, tensor):
    from federated_gcn_aml.federated.messages import FeatureContributionBlock

    return FeatureContributionBlock(
        source_client_id=source,
        recipient_client_id=recipient,
        block_id=block_id,
        row_ids=rows,
        contribution=tensor,
        feature_shape=tuple(tensor.shape),
    )


def test_unsupported_payload_protection_rejected():
    with pytest.raises(NotImplementedError, match="planned but not implemented"):
        parse_args(
            [
                "--federated-mode",
                "fedgcn_feature_pretrain",
                "--num-hops",
                "1",
                "--feature-pretrain-payload-protection",
                "k_filter",
            ]
        )


def test_diagnostic_payload_perturbation_requires_explicit_parameters():
    with pytest.raises(ValueError, match="requires --payload-clip-norm"):
        parse_args(
            [
                "--federated-mode",
                "fedgcn_feature_pretrain",
                "--num-hops",
                "1",
                "--feature-pretrain-payload-protection",
                "central_diagnostic_perturbation",
            ]
        )
    args = parse_args(
        [
            "--federated-mode",
            "fedgcn_feature_pretrain",
            "--num-hops",
            "1",
            "--feature-pretrain-payload-protection",
            "local_diagnostic_perturbation",
            "--payload-clip-norm",
            "1.0",
            "--diagnostic-noise-multiplier",
            "0.0",
            "--seed",
            "42",
        ]
    )
    assert args.diagnostic_seed == 1_000_045


def test_ambiguous_legacy_dp_alias_is_not_accepted():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--federated-mode",
                "fedgcn_feature_pretrain",
                "--num-hops",
                "1",
                "--feature-pretrain-payload-protection",
                "central_dp",
            ]
        )


def test_formal_payload_dp_args_validate_epsilon_delta_and_calibration():
    central_base = [
        "--federated-mode",
        "fedgcn_feature_pretrain",
        "--num-hops",
        "1",
        "--feature-pretrain-payload-protection",
        "central_row_dp",
        "--payload-clip-norm",
        "1.0",
    ]
    with pytest.raises(ValueError, match="requires --payload-dp-epsilon"):
        parse_args(central_base)
    with pytest.raises(ValueError, match="requires --payload-dp-delta"):
        parse_args(central_base + ["--payload-dp-epsilon", "1.0"])
    with pytest.raises(ValueError, match="epsilon <= 1.0"):
        parse_args(central_base + ["--payload-dp-epsilon", "2.0", "--payload-dp-delta", "1e-5"])
    with pytest.raises(ValueError, match="does not accept --diagnostic-noise-multiplier"):
        parse_args(
            central_base
            + [
                "--payload-dp-epsilon",
                "1.0",
                "--payload-dp-delta",
                "1e-5",
                "--diagnostic-noise-multiplier",
                "0.1",
            ]
        )
    args = parse_args(central_base + ["--payload-dp-epsilon", "1.0", "--payload-dp-delta", "1e-5", "--seed", "42"])
    assert args.payload_dp_seed == 1_000_045
    assert args.diagnostic_noise_multiplier is None
    analytic_central = parse_args(
        central_base
        + [
            "--payload-dp-epsilon",
            "2.0",
            "--payload-dp-delta",
            "1e-5",
            "--payload-dp-calibration",
            "analytic_gaussian",
        ]
    )
    assert analytic_central.feature_pretrain_payload_protection == "central_row_dp"
    assert analytic_central.payload_dp_epsilon == 2.0
    assert analytic_central.payload_dp_calibration == "analytic_gaussian"

    analytic_local = parse_args(
        [
            "--federated-mode",
            "fedgcn_feature_pretrain",
            "--num-hops",
            "1",
            "--feature-pretrain-payload-protection",
            "local_row_dp",
            "--payload-clip-norm",
            "1.0",
            "--payload-dp-epsilon",
            "2.0",
            "--payload-dp-delta",
            "1e-5",
            "--payload-dp-calibration",
            "analytic_gaussian",
        ]
    )
    assert analytic_local.feature_pretrain_payload_protection == "local_row_dp"
    assert analytic_local.payload_dp_calibration == "analytic_gaussian"


def test_masked_fedavg_args_resolve_seed_and_min_clients():
    with pytest.raises(ValueError, match="masked_fedavg requires --masked-fedavg-min-clients >= 2"):
        parse_args(["--model-update-aggregation", "masked_fedavg", "--masked-fedavg-min-clients", "1"])
    args = parse_args(["--model-update-aggregation", "masked_fedavg", "--seed", "42"])
    assert args.masked_fedavg_seed == 2_000_045
    assert args.masked_fedavg_min_clients == 2


def _model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def test_plain_fedavg_model_update_message_exposes_local_loss():
    data = _data()
    client = FederatedClient(
        "A",
        data,
        {"in_channels": 2, "hidden_channels": 4, "num_layers": 2, "dropout": 0.0},
        0.01,
        0.0,
        "cpu",
        GCN,
    )
    global_state = _model_state(client.model)
    plan = FedAvgStrategy(model_update_aggregation="plain_fedavg").build_round_plan(
        [("A", client.train_node_count())],
        global_state,
        round_idx=1,
    )
    message = client.train_local_model_update(global_state, local_epochs=1, aggregation_plan=plan)
    assert message.loss is not None
    assert message.state_dict


def test_masked_fedavg_model_update_message_hides_local_loss():
    data_a = _data()
    data_b = _data()
    model_config = {"in_channels": 2, "hidden_channels": 4, "num_layers": 2, "dropout": 0.0}
    client_a = FederatedClient("A", data_a, model_config, 0.01, 0.0, "cpu", GCN, masked_fedavg_seed=99)
    client_b = FederatedClient("B", data_b, model_config, 0.01, 0.0, "cpu", GCN, masked_fedavg_seed=99)
    global_state = _model_state(client_a.model)
    plan = FedAvgStrategy(model_update_aggregation="masked_fedavg").build_round_plan(
        [("A", client_a.train_node_count()), ("B", client_b.train_node_count())],
        global_state,
        round_idx=1,
    )

    message = client_a.train_local_model_update(global_state, local_epochs=1, aggregation_plan=plan)
    assert message.loss is None
    assert message.state_dict == {}
    assert message.masked_update_vector is not None


def test_masked_fedavg_zero_train_message_hides_local_loss():
    data_a = _data()
    data_b = _data()
    data_c = _data()
    data_a.train_mask = torch.zeros_like(data_a.train_mask, dtype=torch.bool)
    model_config = {"in_channels": 2, "hidden_channels": 4, "num_layers": 2, "dropout": 0.0}
    client_a = FederatedClient("A", data_a, model_config, 0.01, 0.0, "cpu", GCN, masked_fedavg_seed=99)
    client_b = FederatedClient("B", data_b, model_config, 0.01, 0.0, "cpu", GCN, masked_fedavg_seed=99)
    client_c = FederatedClient("C", data_c, model_config, 0.01, 0.0, "cpu", GCN, masked_fedavg_seed=99)
    global_state = _model_state(client_a.model)
    plan = FedAvgStrategy(model_update_aggregation="masked_fedavg").build_round_plan(
        [
            ("A", client_a.train_node_count()),
            ("B", client_b.train_node_count()),
            ("C", client_c.train_node_count()),
        ],
        global_state,
        round_idx=1,
    )

    message = client_a.train_local_model_update(global_state, local_epochs=1, aggregation_plan=plan)
    assert message.loss is None
    assert message.masked_update_vector is None


def test_he_transport_roundtrip_if_tenseal_available():
    pytest.importorskip("tenseal")
    transport = TenSEALCKKSTransport()
    block = BlockPlan("A", "A:0", torch.tensor([0]), (1, 2), 0, 1)
    encoded = transport.encode_contribution_block(
        client_block("A", "A", "A:0", block.row_ids, torch.tensor([[1.25, -0.5]]))
    )
    assert encoded.is_encrypted
    assert isinstance(encoded.payload, bytes)
    assert not transport.server_can_decrypt()
    payload = transport.aggregate_blocks([encoded], block)
    decoded = transport.decode_payload_block(payload)
    assert torch.allclose(decoded, torch.tensor([[1.25, -0.5]]), atol=1e-3)


def test_he_protocol_decrypted_payload_matches_plaintext_oracle():
    pytest.importorskip("tenseal")
    data = _data()
    partition = _partition()
    expected = _expected_px(data)
    for hops in (1, 2):
        graphs = build_client_graphs(data, partition, "fedgcn_feature_pretrain", num_hops=hops)
        clients = [
            FederatedClient(
                g.client_id,
                g.data,
                {"in_channels": 2, "hidden_channels": 4, "num_layers": 2},
                0.01,
                0.0,
                "cpu",
                PreAggregatedGCN,
            )
            for g in graphs
        ]
        protocol = FedGCNFeaturePretrainProtocol(num_hops=hops, feature_pretrain_transport="he", block_rows=2)
        protocol.prepare(data, partition, graphs)
        protocol.aggregate_on_server(protocol.collect_contributions(clients))
        communication = protocol.diagnostics()["communication"]["metadata"]
        assert communication["ckks_payload_error_sampled"] is False
        assert communication["ckks_payload_max_abs_error"] is not None
        assert communication["ckks_payload_mean_abs_error"] is not None
        assert communication["ckks_payload_max_abs_error"] <= 1e-3
        assert communication["ckks_payload_mean_abs_error"] <= 1e-4
        for client in clients:
            payload = protocol.build_payload_for_client(client.client_id)
            assert all(block.is_encrypted and isinstance(block.payload, bytes) for block in payload)
            protocol.install_payload(client, payload)
            assert torch.allclose(
                client.data.x.cpu(),
                expected[client.data.global_node_indices.cpu()],
                atol=1e-3,
            )
