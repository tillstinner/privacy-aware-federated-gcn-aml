import pytest
import torch

from federated_gcn_aml.federated.messages import ClientModelUpdateMessage, ClientUpdate
from federated_gcn_aml.federated.strategies import (
    FedAvgStrategy,
    build_masked_update_vector,
    extract_non_floating_state,
    flatten_weighted_delta,
)


def _state(value, counter=None):
    if isinstance(value, torch.Tensor):
        tensor = value.to(torch.float32)
    else:
        tensor = torch.tensor([value], dtype=torch.float32)
    state = {"w": tensor}
    if counter is not None:
        state["counter"] = torch.tensor(counter, dtype=torch.long)
    return state


def _update(client_id, weight, value, counter=None):
    return ClientUpdate(client_id, _state(value, counter), weight, loss=None)


def _masked_messages(strategy, current_state, local_states, train_counts, round_idx=1, masked_seed=123):
    plan = strategy.build_round_plan(
        [(client_id, train_counts[client_id]) for client_id in sorted(local_states)],
        current_state,
        round_idx,
    )
    messages = []
    for client_id in sorted(local_states):
        train_count = int(train_counts[client_id])
        if train_count <= 0 or client_id not in plan.participant_client_ids:
            messages.append(ClientModelUpdateMessage(client_id, train_count, loss=None))
            continue
        weighted_delta = flatten_weighted_delta(
            local_states[client_id],
            current_state,
            plan.tensor_specs,
            plan.weights[client_id],
        )
        messages.append(
            ClientModelUpdateMessage(
                client_id=client_id,
                train_node_count=train_count,
                loss=None,
                masked_update_vector=build_masked_update_vector(
                    client_id=client_id,
                    weighted_delta=weighted_delta,
                    plan=plan,
                    masked_fedavg_seed=masked_seed,
                ),
                non_floating_state_dict=extract_non_floating_state(local_states[client_id], plan.tensor_specs),
            )
        )
    return plan, messages


def _plain_aggregate(weighting, local_states, train_counts):
    return FedAvgStrategy(weighting).aggregate(
        [
            ClientUpdate(client_id, local_states[client_id], train_counts[client_id], loss=None)
            for client_id in sorted(local_states)
        ]
    )


def test_train_node_weighted_fedavg():
    out = FedAvgStrategy("train_nodes").aggregate([_update("a", 1, 1.0), _update("b", 3, 5.0)])
    assert torch.allclose(out["w"], torch.tensor([4.0]))


def test_equal_fedavg():
    out = FedAvgStrategy("equal").aggregate([_update("a", 1, 1.0), _update("b", 3, 5.0)])
    assert torch.allclose(out["w"], torch.tensor([3.0]))


def test_non_floating_buffers_must_match():
    out = FedAvgStrategy("equal").aggregate([_update("a", 1, 1.0, 7), _update("b", 1, 5.0, 7)])
    assert out["counter"].item() == 7
    with pytest.raises(ValueError, match="Non-floating"):
        FedAvgStrategy("equal").aggregate([_update("a", 1, 1.0, 7), _update("b", 1, 5.0, 8)])


def test_masked_pairwise_masks_cancel_for_three_clients():
    current = {"w": torch.tensor([0.0, 0.0], dtype=torch.float32)}
    states = {
        "a": _state(torch.tensor([1.0, 2.0])),
        "b": _state(torch.tensor([3.0, 5.0])),
        "c": _state(torch.tensor([-1.0, 4.0])),
    }
    counts = {"a": 1, "b": 1, "c": 1}
    strategy = FedAvgStrategy("equal", model_update_aggregation="masked_fedavg")
    plan, messages = _masked_messages(strategy, current, states, counts, masked_seed=123)

    masked_sum = sum(message.masked_update_vector for message in messages)
    unmasked_sum = sum(
        flatten_weighted_delta(states[client_id], current, plan.tensor_specs, plan.weights[client_id])
        for client_id in plan.participant_client_ids
    )
    assert torch.allclose(masked_sum, unmasked_sum)


@pytest.mark.parametrize("weighting", ["equal", "train_nodes"])
def test_masked_fedavg_matches_plain_fedavg(weighting):
    current = {"w": torch.tensor([1.0, -2.0], dtype=torch.float32)}
    states = {
        "a": _state(torch.tensor([3.0, 0.0])),
        "b": _state(torch.tensor([7.0, -1.0])),
        "c": _state(torch.tensor([-5.0, 4.0])),
    }
    counts = {"a": 1, "b": 3, "c": 2}
    masked = FedAvgStrategy(weighting, model_update_aggregation="masked_fedavg")
    plan, messages = _masked_messages(masked, current, states, counts, masked_seed=321)
    out = masked.aggregate_model_updates(messages, current_state=current, round_plan=plan)
    plain = _plain_aggregate(weighting, states, counts)
    assert torch.allclose(out["w"], plain["w"], atol=1e-6)


def test_masked_fedavg_ignores_zero_train_clients_like_plain_fedavg():
    current = {"w": torch.tensor([0.0], dtype=torch.float32)}
    states = {"a": _state(1.0), "b": _state(5.0), "c": _state(999.0)}
    counts = {"a": 1, "b": 1, "c": 0}
    masked = FedAvgStrategy("train_nodes", model_update_aggregation="masked_fedavg")
    plan, messages = _masked_messages(masked, current, states, counts, masked_seed=55)
    out = masked.aggregate_model_updates(messages, current_state=current, round_plan=plan)
    plain = _plain_aggregate("train_nodes", states, counts)
    assert plan.participant_client_ids == ("a", "b")
    assert torch.allclose(out["w"], plain["w"], atol=1e-6)


def test_masked_fedavg_messages_do_not_contain_plain_floating_states():
    current = {"w": torch.tensor([0.0], dtype=torch.float32)}
    states = {"a": _state(1.0), "b": _state(3.0)}
    counts = {"a": 1, "b": 1}
    strategy = FedAvgStrategy("equal", model_update_aggregation="masked_fedavg")
    _, messages = _masked_messages(strategy, current, states, counts, masked_seed=7)
    for message in messages:
        assert message.state_dict == {}
        assert message.masked_update_vector is not None


def test_masked_fedavg_aggregate_uses_masked_vectors_not_plain_state_dicts():
    current = {"w": torch.tensor([0.0], dtype=torch.float32)}
    states = {"a": _state(1.0), "b": _state(3.0)}
    counts = {"a": 1, "b": 1}
    strategy = FedAvgStrategy("equal", model_update_aggregation="masked_fedavg")
    plan, messages = _masked_messages(strategy, current, states, counts, masked_seed=7)
    poisoned_messages = [
        ClientModelUpdateMessage(
            client_id=message.client_id,
            train_node_count=message.train_node_count,
            loss=message.loss,
            state_dict={"w": torch.tensor([999.0], dtype=torch.float32)},
            masked_update_vector=message.masked_update_vector,
            non_floating_state_dict=message.non_floating_state_dict,
        )
        for message in messages
    ]
    out = strategy.aggregate_model_updates(poisoned_messages, current_state=current, round_plan=plan)
    plain = _plain_aggregate("equal", states, counts)
    assert torch.allclose(out["w"], plain["w"], atol=1e-6)


def test_masked_fedavg_deterministic_by_seed_and_round():
    current = {"w": torch.tensor([0.0, 0.0], dtype=torch.float32)}
    states = {"a": _state(torch.tensor([1.0, 2.0])), "b": _state(torch.tensor([5.0, 7.0]))}
    counts = {"a": 1, "b": 1}
    strategy = FedAvgStrategy("equal", model_update_aggregation="masked_fedavg")
    plan1, messages1 = _masked_messages(strategy, current, states, counts, round_idx=4, masked_seed=91)
    _, messages2 = _masked_messages(strategy, current, states, counts, round_idx=4, masked_seed=91)
    plan3, messages3 = _masked_messages(strategy, current, states, counts, round_idx=5, masked_seed=91)
    _, messages4 = _masked_messages(strategy, current, states, counts, round_idx=4, masked_seed=92)

    assert torch.equal(messages1[0].masked_update_vector, messages2[0].masked_update_vector)
    assert not torch.equal(messages1[0].masked_update_vector, messages3[0].masked_update_vector)
    assert not torch.equal(messages1[0].masked_update_vector, messages4[0].masked_update_vector)
    out1 = strategy.aggregate_model_updates(messages1, current_state=current, round_plan=plan1)
    out3 = strategy.aggregate_model_updates(messages3, current_state=current, round_plan=plan3)
    plain = _plain_aggregate("equal", states, counts)
    assert torch.allclose(out1["w"], plain["w"], atol=1e-6)
    assert torch.allclose(out3["w"], plain["w"], atol=1e-6)


def test_masked_fedavg_missing_client_raises_dropout_unsupported_error():
    current = {"w": torch.tensor([0.0], dtype=torch.float32)}
    states = {"a": _state(1.0), "b": _state(2.0)}
    counts = {"a": 1, "b": 1}
    strategy = FedAvgStrategy("equal", model_update_aggregation="masked_fedavg")
    plan, messages = _masked_messages(strategy, current, states, counts, masked_seed=8)
    with pytest.raises(RuntimeError, match="dropout recovery is not implemented"):
        strategy.aggregate_model_updates(messages[:1], current_state=current, round_plan=plan)


def test_masked_fedavg_non_floating_buffers_must_match():
    current = {"w": torch.tensor([0.0], dtype=torch.float32), "counter": torch.tensor(7, dtype=torch.long)}
    states = {"a": _state(1.0, 7), "b": _state(3.0, 7)}
    counts = {"a": 1, "b": 1}
    strategy = FedAvgStrategy("equal", model_update_aggregation="masked_fedavg")
    plan, messages = _masked_messages(strategy, current, states, counts, masked_seed=6)
    out = strategy.aggregate_model_updates(messages, current_state=current, round_plan=plan)
    assert out["counter"].item() == 7

    bad_states = {"a": _state(1.0, 7), "b": _state(3.0, 8)}
    bad_plan, bad_messages = _masked_messages(strategy, current, bad_states, counts, masked_seed=6)
    with pytest.raises(ValueError, match="Non-floating"):
        strategy.aggregate_model_updates(bad_messages, current_state=current, round_plan=bad_plan)


def test_masked_fedavg_diagnostics_record_secure_aggregation_fields():
    current = {"w": torch.tensor([0.0], dtype=torch.float32)}
    states = {"a": _state(1.0), "b": _state(3.0)}
    counts = {"a": 1, "b": 2}
    strategy = FedAvgStrategy(
        "train_nodes",
        model_update_aggregation="masked_fedavg",
        masked_fedavg_min_clients=2,
    )
    plan, messages = _masked_messages(strategy, current, states, counts, masked_seed=17)
    strategy.aggregate_model_updates(messages, current_state=current, round_plan=plan)
    diagnostics = strategy.diagnostics()
    assert diagnostics["secure_aggregation"] is True
    assert diagnostics["secure_aggregation_protocol"] == "secagg_style_pairwise_masking"
    assert diagnostics["secure_aggregation_claim"] == "additive_masking_no_dropout"
    assert diagnostics["model_updates_visible_to_server"] is False
    assert diagnostics["plaintext_model_updates_visible_to_server"] is False
    assert diagnostics["masked_model_updates_visible_to_server"] is True
    assert diagnostics["aggregate_update_visible_to_server"] is True
    assert diagnostics["masked_fedavg_rounds"] == 1
    assert diagnostics["masked_fedavg_valid_clients_min"] == 2
    assert diagnostics["masked_fedavg_vector_numel"] == 1
