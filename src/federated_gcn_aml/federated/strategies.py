from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import torch

from federated_gcn_aml.federated.messages import (
    ClientModelUpdateMessage,
    ClientUpdate,
    ModelTensorSpec,
    ModelUpdateAggregationPlan,
)


def _validate_weighting(weighting: str) -> None:
    if weighting not in {"train_nodes", "equal"}:
        raise ValueError(f"Unsupported FedAvg weighting: {weighting}")


def _valid_client_weights(
    client_train_counts: list[tuple[str, int]],
    weighting: str,
) -> dict[str, float]:
    _validate_weighting(weighting)
    valid = [(client_id, int(train_count)) for client_id, train_count in client_train_counts if int(train_count) > 0]
    if not valid:
        raise ValueError("No clients had training nodes; cannot aggregate model states.")
    if weighting == "train_nodes":
        total = float(sum(train_count for _, train_count in valid))
        return {client_id: float(train_count) / total for client_id, train_count in valid}
    weight = 1.0 / float(len(valid))
    return {client_id: weight for client_id, _ in valid}


def floating_tensor_specs(state_dict: dict[str, torch.Tensor]) -> tuple[ModelTensorSpec, ...]:
    specs: list[ModelTensorSpec] = []
    for key, tensor in state_dict.items():
        if tensor.is_complex():
            raise ValueError(f"masked_fedavg does not support complex tensor {key!r} in v1.")
        if tensor.is_floating_point():
            detached = tensor.detach().cpu()
            specs.append(
                ModelTensorSpec(
                    key=key,
                    shape=tuple(detached.shape),
                    dtype=detached.dtype,
                    numel=int(detached.numel()),
                )
            )
    return tuple(specs)


def flatten_weighted_delta(
    local_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    tensor_specs: tuple[ModelTensorSpec, ...],
    weight: float,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for spec in tensor_specs:
        local = local_state[spec.key].detach().cpu().to(torch.float64)
        current = current_state[spec.key].detach().cpu().to(torch.float64)
        parts.append(((local - current) * float(weight)).reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(parts).to(torch.float64)


def extract_non_floating_state(
    state_dict: dict[str, torch.Tensor],
    tensor_specs: tuple[ModelTensorSpec, ...],
) -> dict[str, torch.Tensor]:
    floating_keys = {spec.key for spec in tensor_specs}
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if key not in floating_keys
    }


def unflatten_aggregate_delta(
    current_state: dict[str, torch.Tensor],
    aggregate_delta: torch.Tensor,
    tensor_specs: tuple[ModelTensorSpec, ...],
) -> dict[str, torch.Tensor]:
    out = {key: value.detach().cpu().clone() for key, value in current_state.items()}
    offset = 0
    aggregate_delta = aggregate_delta.detach().cpu().to(torch.float64)
    for spec in tensor_specs:
        chunk = aggregate_delta[offset : offset + spec.numel].reshape(spec.shape)
        out[spec.key] = (current_state[spec.key].detach().cpu().to(torch.float64) + chunk).to(spec.dtype)
        offset += spec.numel
    if offset != int(aggregate_delta.numel()):
        raise ValueError("Aggregate delta vector length did not match model tensor specification.")
    return out


def _pairwise_mask_seed(
    masked_fedavg_seed: int,
    round_idx: int,
    lower_client_id: str,
    higher_client_id: str,
    vector_numel: int,
) -> int:
    material = f"{masked_fedavg_seed}|{round_idx}|{lower_client_id}|{higher_client_id}|{vector_numel}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def pairwise_mask(
    *,
    masked_fedavg_seed: int,
    round_idx: int,
    lower_client_id: str,
    higher_client_id: str,
    vector_numel: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        _pairwise_mask_seed(
            masked_fedavg_seed=masked_fedavg_seed,
            round_idx=round_idx,
            lower_client_id=lower_client_id,
            higher_client_id=higher_client_id,
            vector_numel=vector_numel,
        )
    )
    return torch.randn(vector_numel, generator=generator, dtype=torch.float64)


def build_masked_update_vector(
    *,
    client_id: str,
    weighted_delta: torch.Tensor,
    plan: ModelUpdateAggregationPlan,
    masked_fedavg_seed: int,
) -> torch.Tensor:
    if plan.mode != "masked_fedavg":
        raise ValueError("Masked update vectors can only be built for masked_fedavg plans.")
    if client_id not in plan.participant_client_ids:
        return torch.empty(0, dtype=torch.float64)

    masked = weighted_delta.detach().cpu().to(torch.float64).clone()
    if int(masked.numel()) != int(plan.vector_numel):
        raise ValueError(
            f"Client {client_id} weighted delta has {masked.numel()} elements, expected {plan.vector_numel}."
        )
    for peer_id in plan.participant_client_ids:
        if peer_id == client_id:
            continue
        lower, higher = sorted((client_id, peer_id))
        mask = pairwise_mask(
            masked_fedavg_seed=int(masked_fedavg_seed),
            round_idx=int(plan.round_idx),
            lower_client_id=lower,
            higher_client_id=higher,
            vector_numel=int(plan.vector_numel),
        )
        if client_id == lower:
            masked += mask
        else:
            masked -= mask
    return masked


@dataclass
class FedAvgStrategy:
    weighting: str = "train_nodes"
    model_update_aggregation: str = "plain_fedavg"
    masked_fedavg_min_clients: int = 2
    _rounds: int = field(default=0, init=False)
    _valid_clients_min: int | None = field(default=None, init=False)
    _valid_clients_max: int | None = field(default=None, init=False)
    _vector_numel: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _validate_weighting(self.weighting)
        if self.model_update_aggregation not in {"plain_fedavg", "masked_fedavg"}:
            raise ValueError(f"Unsupported model update aggregation: {self.model_update_aggregation}")
        if self.masked_fedavg_min_clients < 1:
            raise ValueError("masked_fedavg_min_clients must be positive.")
        if self.model_update_aggregation == "masked_fedavg" and self.masked_fedavg_min_clients < 2:
            raise ValueError("masked_fedavg requires at least two valid clients.")

    def aggregate(self, updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
        valid = [update for update in updates if update.train_node_count > 0]
        if not valid:
            raise ValueError("No clients had training nodes; cannot aggregate model states.")
        weights = _valid_client_weights(
            [(update.client_id, update.train_node_count) for update in valid],
            self.weighting,
        )

        aggregated: dict[str, torch.Tensor] = {}
        for key, first in valid[0].state_dict.items():
            tensors = [update.state_dict[key].detach().cpu() for update in valid]
            if first.is_floating_point() or first.is_complex():
                stacked = torch.stack([tensor.to(torch.float64) for tensor in tensors])
                weight_vector = torch.tensor([weights[update.client_id] for update in valid], dtype=torch.float64)
                view_shape = (len(valid),) + (1,) * (stacked.dim() - 1)
                averaged = (stacked * weight_vector.view(view_shape)).sum(dim=0)
                aggregated[key] = averaged.to(dtype=first.dtype)
            else:
                for tensor in tensors[1:]:
                    if not torch.equal(tensors[0], tensor):
                        raise ValueError(f"Non-floating state buffer {key!r} differs across clients.")
                aggregated[key] = tensors[0].clone()
        return aggregated

    def build_round_plan(
        self,
        client_train_counts: list[tuple[str, int]],
        current_state: dict[str, torch.Tensor],
        round_idx: int,
    ) -> ModelUpdateAggregationPlan:
        weights = _valid_client_weights(client_train_counts, self.weighting)
        participants = tuple(sorted(weights))
        tensor_specs = (
            floating_tensor_specs(current_state)
            if self.model_update_aggregation == "masked_fedavg"
            else tuple()
        )
        vector_numel = sum(spec.numel for spec in tensor_specs)
        if self.model_update_aggregation == "masked_fedavg" and len(participants) < self.masked_fedavg_min_clients:
            raise RuntimeError(
                f"masked_fedavg requires at least {self.masked_fedavg_min_clients} valid clients, "
                f"but only {len(participants)} had training nodes. Dropout/low-participation recovery is not implemented."
            )
        return ModelUpdateAggregationPlan(
            round_idx=int(round_idx),
            mode=self.model_update_aggregation,
            participant_client_ids=participants,
            weights=weights,
            tensor_specs=tensor_specs,
            vector_numel=int(vector_numel),
            masked_fedavg_min_clients=int(self.masked_fedavg_min_clients),
        )

    def aggregate_model_updates(
        self,
        messages: list[ClientModelUpdateMessage],
        *,
        current_state: dict[str, torch.Tensor],
        round_plan: ModelUpdateAggregationPlan,
    ) -> dict[str, torch.Tensor]:
        self._record_round(round_plan)
        if round_plan.mode == "plain_fedavg":
            return self.aggregate(
                [
                    ClientUpdate(
                        client_id=message.client_id,
                        state_dict=message.state_dict,
                        train_node_count=message.train_node_count,
                        loss=message.loss,
                        diagnostics=message.diagnostics,
                    )
                    for message in messages
                ]
            )
        if round_plan.mode == "masked_fedavg":
            return self._aggregate_masked(messages, current_state=current_state, round_plan=round_plan)
        raise ValueError(f"Unsupported model update aggregation plan mode: {round_plan.mode}")

    def _aggregate_masked(
        self,
        messages: list[ClientModelUpdateMessage],
        *,
        current_state: dict[str, torch.Tensor],
        round_plan: ModelUpdateAggregationPlan,
    ) -> dict[str, torch.Tensor]:
        by_id = {message.client_id: message for message in messages}
        missing = [client_id for client_id in round_plan.participant_client_ids if client_id not in by_id]
        if missing:
            raise RuntimeError(
                "masked_fedavg received missing client updates "
                f"{missing}; dropout recovery is not implemented in v1."
            )

        aggregate_delta = torch.zeros(round_plan.vector_numel, dtype=torch.float64)
        for client_id in round_plan.participant_client_ids:
            message = by_id[client_id]
            if message.masked_update_vector is None:
                raise RuntimeError(f"masked_fedavg client {client_id} did not provide a masked update vector.")
            aggregate_delta += message.masked_update_vector.detach().cpu().to(torch.float64)

        new_state = unflatten_aggregate_delta(current_state, aggregate_delta, round_plan.tensor_specs)
        floating_keys = {spec.key for spec in round_plan.tensor_specs}
        non_floating_keys = [key for key in current_state if key not in floating_keys]
        for key in non_floating_keys:
            tensors = []
            for client_id in round_plan.participant_client_ids:
                state = by_id[client_id].non_floating_state_dict
                if key not in state:
                    raise ValueError(f"masked_fedavg client {client_id} is missing non-floating state buffer {key!r}.")
                tensors.append(state[key].detach().cpu())
            for tensor in tensors[1:]:
                if not torch.equal(tensors[0], tensor):
                    raise ValueError(f"Non-floating state buffer {key!r} differs across clients.")
            new_state[key] = tensors[0].clone()
        return new_state

    def _record_round(self, round_plan: ModelUpdateAggregationPlan) -> None:
        valid_count = len(round_plan.participant_client_ids)
        self._rounds += 1
        self._valid_clients_min = valid_count if self._valid_clients_min is None else min(self._valid_clients_min, valid_count)
        self._valid_clients_max = valid_count if self._valid_clients_max is None else max(self._valid_clients_max, valid_count)
        self._vector_numel = int(round_plan.vector_numel)

    def diagnostics(self) -> dict:
        secure = self.model_update_aggregation == "masked_fedavg"
        return {
            "model_update_aggregation": self.model_update_aggregation,
            "secure_aggregation": secure,
            "secure_aggregation_protocol": "secagg_style_pairwise_masking" if secure else None,
            "secure_aggregation_claim": "additive_masking_no_dropout" if secure else None,
            "secure_aggregation_min_clients": int(self.masked_fedavg_min_clients) if secure else None,
            "secure_aggregation_dropout_resilience": False,
            "secure_aggregation_secret_sharing": False,
            "secure_aggregation_finite_field_quantization": False,
            "secure_aggregation_malicious_server_protection": False,
            "secure_aggregation_client_collusion_protection": "non_colluding_clients_assumed" if secure else None,
            "model_updates_visible_to_server": not secure,
            "plaintext_model_updates_visible_to_server": not secure,
            "masked_model_updates_visible_to_server": secure,
            "aggregate_update_visible_to_server": True,
            "model_update_dp": False,
            "masked_fedavg_rounds": int(self._rounds) if secure else 0,
            "masked_fedavg_valid_clients_min": self._valid_clients_min if secure else None,
            "masked_fedavg_valid_clients_max": self._valid_clients_max if secure else None,
            "masked_fedavg_vector_numel": self._vector_numel if secure else None,
            "masked_fedavg_missing_client_policy": "raise_no_dropout_recovery" if secure else None,
            "masked_fedavg_non_floating_state_policy": "validate_matching_client_buffers" if secure else None,
        }
