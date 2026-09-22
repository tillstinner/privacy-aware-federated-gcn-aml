from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import torch

from federated_gcn_aml.federated.messages import BlockPlan, ClientFeatureContribution


DIAGNOSTIC_PERTURBATION_MODES = {"central_diagnostic_perturbation", "local_diagnostic_perturbation"}
FORMAL_ROW_DP_MODES = {"central_row_dp", "local_row_dp"}
UNSUPPORTED_PAYLOAD_PROTECTION_MODES = {"k_filter", "central_row_dp_k_filter"}
SUPPORTED_PAYLOAD_PROTECTION_MODES = (
    {"none"}
    | DIAGNOSTIC_PERTURBATION_MODES
    | FORMAL_ROW_DP_MODES
    | UNSUPPORTED_PAYLOAD_PROTECTION_MODES
)
NOT_CLAIMED_BY_ROW_DP = [
    "account_level_dp",
    "transaction_level_dp",
    "edge_topology_dp",
    "client_level_dp",
    "model_update_dp",
    "end_to_end_dp",
]

def classical_gaussian_sigma(*, epsilon: float, delta: float, sensitivity_l2: float) -> float:
    if not math.isfinite(epsilon) or not math.isfinite(delta) or not math.isfinite(sensitivity_l2):
        raise ValueError("classical Gaussian payload DP parameters must be finite.")
    if epsilon <= 0:
        raise ValueError("payload DP epsilon must be positive.")
    if epsilon > 1.0:
        raise ValueError("classical Gaussian payload DP calibration requires epsilon <= 1.0.")
    if delta <= 0 or delta >= 1:
        raise ValueError("payload DP delta must be in (0, 1).")
    if sensitivity_l2 <= 0:
        raise ValueError("payload DP sensitivity must be positive.")
    return float(math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity_l2 / epsilon)


def _standard_normal_cdf(value: float) -> float:
    # Avoid cancellation of tiny lower-tail probabilities at large epsilon.
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def analytic_gaussian_sigma(
    *,
    epsilon: float,
    delta: float,
    sensitivity_l2: float,
    tol: float = 1e-12,
    max_iterations: int = 10_000,
) -> float:
    """Calibrate Gaussian noise using the Balle-Wang analytic mechanism.

    The implementation follows the scalar root-finding formulation used by the
    analytic Gaussian mechanism: locate the zero of B+ or B- by interval
    doubling, then bisection. It is intentionally dependency-free so formal
    payload-DP experiments do not require diffprivlib.
    """

    if not math.isfinite(epsilon) or not math.isfinite(delta) or not math.isfinite(sensitivity_l2):
        raise ValueError("analytic Gaussian payload DP parameters must be finite.")
    if epsilon <= 0:
        raise ValueError("payload DP epsilon must be positive.")
    if delta <= 0 or delta >= 1:
        raise ValueError("payload DP delta must be in (0, 1).")
    if sensitivity_l2 <= 0:
        raise ValueError("payload DP sensitivity must be positive.")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("analytic Gaussian tolerance must be positive.")

    try:
        exp_epsilon = math.exp(epsilon)
    except OverflowError as exc:
        raise ValueError(
            "analytic Gaussian calibration failed because epsilon is too large for stable "
            "floating-point evaluation."
        ) from exc

    def b_plus(value: float) -> float:
        return (
            _standard_normal_cdf(math.sqrt(epsilon * value))
            - exp_epsilon * _standard_normal_cdf(-math.sqrt(epsilon * (value + 2.0)))
            - delta
        )

    def b_minus(value: float) -> float:
        return (
            _standard_normal_cdf(-math.sqrt(epsilon * value))
            - exp_epsilon * _standard_normal_cdf(-math.sqrt(epsilon * (value + 2.0)))
            - delta
        )

    delta_0 = b_plus(0.0)
    target = b_plus if delta_0 < 0.0 else b_minus

    left = 0.0
    right = 1.0
    f_left = target(left)
    f_right = target(right)
    interval_iterations = 0
    while f_left * f_right > 0.0:
        left = right
        f_left = f_right
        right *= 2.0
        f_right = target(right)
        interval_iterations += 1
        if interval_iterations > max_iterations or not math.isfinite(f_right):
            raise ValueError("analytic Gaussian calibration failed to bracket the root.")

    for _ in range(max_iterations):
        if (right - left) <= tol * max(1.0, right):
            break
        middle = (left + right) / 2.0
        f_middle = target(middle)
        if f_middle * f_left <= 0.0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    else:
        raise ValueError("analytic Gaussian calibration failed to converge.")

    # Algorithm 1 uses root/2. Keep the feasible endpoint of the bracket.
    root = left if delta_0 < 0.0 else right
    alpha = math.sqrt(1.0 + root / 2.0) + math.sqrt(root / 2.0)
    if delta_0 < 0.0:
        # Equivalent to subtracting the two roots, without cancellation.
        alpha = 1.0 / alpha
    sigma = alpha * sensitivity_l2 / math.sqrt(2.0 * epsilon)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("analytic Gaussian calibration returned a non-positive or non-finite sigma.")
    return float(sigma)


class PayloadProtectionPolicy:
    """Hook points for payload-release protections."""

    name = "base"
    requested_name: str | None = None

    def protect_compact_contribution_before_blocking(
        self,
        contribution: ClientFeatureContribution,
        metadata: dict[str, Any] | None = None,
    ) -> ClientFeatureContribution:
        return contribution

    def protect_source_contribution_before_transport(self, block):
        return block

    def filter_recipient_rows(self, row_plan, metadata: dict[str, Any] | None = None):
        return row_plan

    def protect_aggregate_before_delivery(
        self,
        aggregate,
        *,
        transport=None,
        block_plan: BlockPlan | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        return aggregate

    def plaintext_aggregate_delta_for_block(self, block_plan: BlockPlan) -> torch.Tensor | None:
        return None

    def protect_decrypted_payload_after_client_decode(self, tensor, metadata: dict[str, Any] | None = None):
        return tensor

    def install_fallback_for_suppressed_rows(self, tensor, metadata: dict[str, Any] | None = None):
        return tensor

    def record_block_release(self, source_client_id: str, row_ids: torch.Tensor) -> None:
        return None

    def enable_plaintext_oracle_diagnostics(self) -> bool:
        return True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "payload_protection_policy": self.name,
            "payload_protection_policy_requested": self.requested_name or self.name,
            "rows_suppressed": 0,
            "rows_noised": 0,
            "support_eq_1_rows": None,
            "support_lt_k_rows": None,
            "differential_privacy": False,
            "formal_differential_privacy": False,
            "formal_dp_accounting": False,
            "diagnostic_perturbation": False,
            "diagnostic_perturbation_mechanism": None,
            "diagnostic_perturbation_scope": None,
            "diagnostic_noise_multiplier": None,
            "diagnostic_noise_std": None,
            "diagnostic_seed": None,
            "payload_protection_kind": "none",
            "payload_clip_norm": None,
            "payload_dp_claim": None,
            "claim_scope": None,
            "privacy_unit": None,
            "adjacency": None,
            "payload_dp_mechanism": None,
            "payload_dp_scope": None,
            "payload_dp_clip_norm": None,
            "payload_dp_sensitivity_l2": None,
            "payload_dp_noise_multiplier": None,
            "payload_dp_noise_std": None,
            "payload_dp_seed": None,
            "payload_dp_epsilon": None,
            "payload_dp_delta": None,
            "payload_dp_calibration": None,
            "dp_epsilon": None,
            "dp_delta": None,
            "epsilon_per_release": None,
            "delta_per_release": None,
            "epsilon_total": None,
            "delta_total": None,
            "composition": None,
            "num_noised_releases": 0,
            "num_unique_private_units": 0,
            "max_release_count_per_unit": 0,
            "duplicate_release_policy": None,
            "trust_boundary": None,
            "not_claimed": [],
            "formal_dp_oracle_diagnostics_enabled": True,
            "diagnostics_scope": "non_dp_diagnostics",
            "rows_clipped": 0,
            "clip_fraction": 0.0,
            "noise_l2_norm_total": 0.0,
            "noise_l2_norm_mean": 0.0,
        }


class NoPayloadProtection(PayloadProtectionPolicy):
    name = "none"


class _GaussianPayloadProtection(PayloadProtectionPolicy):
    def __init__(
        self,
        *,
        clip_norm: float,
        noise_std: float,
        random_seed: int,
        name: str,
        protection_scope: str,
        noise_multiplier: float | None,
        formal_differential_privacy: bool,
        payload_protection_kind: str,
        payload_dp_claim: str | None,
        claim_scope: str | None,
        privacy_unit: str | None,
        adjacency: str | None,
        sensitivity_l2: float | None,
        epsilon: float | None,
        delta: float | None,
        calibration: str | None,
        trust_boundary: str | None,
        duplicate_release_policy: str | None,
        composition: str | None,
        requested_name: str | None = None,
    ) -> None:
        if clip_norm <= 0:
            raise ValueError("payload clip norm must be positive.")
        if noise_std < 0:
            raise ValueError("payload noise std must be non-negative.")
        self.clip_norm = float(clip_norm)
        self.noise_std = float(noise_std)
        self.noise_multiplier = None if noise_multiplier is None else float(noise_multiplier)
        self.random_seed = int(random_seed)
        self.name = name
        self.requested_name = requested_name or name
        self.protection_scope = protection_scope
        self.formal_differential_privacy = bool(formal_differential_privacy)
        self.payload_protection_kind = payload_protection_kind
        self.payload_dp_claim = payload_dp_claim
        self.claim_scope = claim_scope
        self.privacy_unit = privacy_unit
        self.adjacency = adjacency
        self.sensitivity_l2 = sensitivity_l2
        self.epsilon = epsilon
        self.delta = delta
        self.calibration = calibration
        self.trust_boundary = trust_boundary
        self.duplicate_release_policy = duplicate_release_policy
        self.composition = composition
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.random_seed)
        self._rows_seen = 0
        self._rows_clipped = 0
        self._rows_noised = 0
        self._noise_l2_norm_total = 0.0

    def _clip_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().cpu()
        if not tensor.numel():
            return tensor.clone()
        norms = tensor.norm(p=2, dim=1)
        scale = torch.ones_like(norms)
        clipped = norms > self.clip_norm
        scale[clipped] = self.clip_norm / norms[clipped].clamp_min(1e-12)
        self._rows_seen += int(norms.numel())
        self._rows_clipped += int(clipped.sum().item())
        return tensor * scale.view(-1, 1)

    def _sample_noise(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float) -> torch.Tensor:
        if self.noise_std == 0.0 or not shape or int(torch.tensor(shape).prod().item()) == 0:
            return torch.zeros(shape, dtype=dtype)
        noise = torch.randn(shape, generator=self._generator, dtype=dtype) * self.noise_std
        if len(shape) >= 2:
            self._rows_noised += int(shape[0])
            self._noise_l2_norm_total += float(noise.reshape(shape[0], -1).norm(p=2, dim=1).sum().item())
        return noise

    def _noise_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._sample_noise(tuple(tensor.shape), dtype=tensor.dtype)

    def _release_count_metadata(self) -> dict[str, Any]:
        return {
            "num_noised_releases": self._rows_noised,
            "num_unique_private_units": self._rows_noised,
            "max_release_count_per_unit": 1 if self._rows_noised else 0,
        }

    def _base_diagnostics(self) -> dict[str, Any]:
        clip_fraction = float(self._rows_clipped / self._rows_seen) if self._rows_seen else 0.0
        release_counts = self._release_count_metadata()
        formal = self.formal_differential_privacy
        diagnostic = self.payload_protection_kind == "diagnostic_gaussian_perturbation"
        return {
            "payload_protection_policy": self.name,
            "payload_protection_policy_requested": self.requested_name,
            "rows_suppressed": 0,
            "rows_noised": self._rows_noised,
            "support_eq_1_rows": None,
            "support_lt_k_rows": None,
            "differential_privacy": formal,
            "formal_differential_privacy": formal,
            "formal_dp_accounting": formal,
            "diagnostic_perturbation": diagnostic,
            "diagnostic_perturbation_mechanism": "gaussian" if diagnostic else None,
            "diagnostic_perturbation_scope": self.protection_scope if diagnostic else None,
            "diagnostic_noise_multiplier": self.noise_multiplier if diagnostic else None,
            "diagnostic_noise_std": float(self.noise_std) if diagnostic else None,
            "diagnostic_seed": self.random_seed if diagnostic else None,
            "payload_protection_kind": self.payload_protection_kind,
            "payload_clip_norm": float(self.clip_norm),
            "payload_dp_claim": self.payload_dp_claim if formal else None,
            "claim_scope": self.claim_scope,
            "privacy_unit": self.privacy_unit,
            "adjacency": self.adjacency,
            "payload_dp_mechanism": "gaussian" if formal else None,
            "payload_dp_scope": self.protection_scope if formal else None,
            "payload_dp_clip_norm": float(self.clip_norm) if formal else None,
            "payload_dp_sensitivity_l2": self.sensitivity_l2 if formal else None,
            "payload_dp_noise_multiplier": None,
            "payload_dp_noise_std": float(self.noise_std) if formal else None,
            "payload_dp_seed": self.random_seed if formal else None,
            "payload_dp_epsilon": self.epsilon if formal else None,
            "payload_dp_delta": self.delta if formal else None,
            "payload_dp_calibration": self.calibration if formal else None,
            "dp_epsilon": self.epsilon if formal else None,
            "dp_delta": self.delta if formal else None,
            "epsilon_per_release": self.epsilon if formal else None,
            "delta_per_release": self.delta if formal else None,
            "epsilon_total": self.epsilon if formal else None,
            "delta_total": self.delta if formal else None,
            "composition": self.composition,
            **release_counts,
            "duplicate_release_policy": self.duplicate_release_policy,
            "trust_boundary": self.trust_boundary,
            "not_claimed": list(NOT_CLAIMED_BY_ROW_DP) if formal else [],
            "formal_dp_oracle_diagnostics_enabled": self.enable_plaintext_oracle_diagnostics(),
            "diagnostics_scope": (
                "no_clean_payload_oracle_for_formal_dp"
                if formal
                else "diagnostic_gaussian_perturbation_diagnostics"
            ),
            "rows_clipped": self._rows_clipped,
            "clip_fraction": clip_fraction,
            "noise_l2_norm_total": self._noise_l2_norm_total,
            "noise_l2_norm_mean": (
                float(self._noise_l2_norm_total / self._rows_noised) if self._rows_noised else 0.0
            ),
        }

    def enable_plaintext_oracle_diagnostics(self) -> bool:
        return not self.formal_differential_privacy

    def diagnostics(self) -> dict[str, Any]:
        return self._base_diagnostics()


class CentralDiagnosticPerturbation(_GaussianPayloadProtection):
    def __init__(
        self,
        clip_norm: float,
        noise_multiplier: float,
        diagnostic_seed: int,
    ) -> None:
        if noise_multiplier < 0:
            raise ValueError("diagnostic noise multiplier must be non-negative.")
        super().__init__(
            clip_norm=clip_norm,
            noise_std=float(clip_norm * noise_multiplier),
            noise_multiplier=noise_multiplier,
            random_seed=diagnostic_seed,
            name="central_diagnostic_perturbation",
            protection_scope="central_aggregate",
            formal_differential_privacy=False,
            payload_protection_kind="diagnostic_gaussian_perturbation",
            payload_dp_claim=None,
            claim_scope=None,
            privacy_unit=None,
            adjacency=None,
            sensitivity_l2=None,
            epsilon=None,
            delta=None,
            calibration=None,
            trust_boundary=None,
            duplicate_release_policy=None,
            composition=None,
        )
        self._central_noise_by_key: dict[tuple[str, str], torch.Tensor] = {}

    def protect_compact_contribution_before_blocking(
        self,
        contribution: ClientFeatureContribution,
        metadata: dict[str, Any] | None = None,
    ) -> ClientFeatureContribution:
        clipped = self._clip_rows(contribution.contribution)
        return replace(contribution, contribution=clipped)

    def protect_aggregate_before_delivery(
        self,
        aggregate,
        *,
        transport=None,
        block_plan: BlockPlan | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if transport is None or block_plan is None:
            raise ValueError("central_diagnostic_perturbation aggregate protection requires transport and block_plan.")
        noise = self._noise_like(torch.zeros(block_plan.shape, dtype=torch.float))
        self._central_noise_by_key[(block_plan.recipient_client_id, block_plan.block_id)] = noise.clone()
        return transport.add_plaintext_to_payload_block(aggregate, noise, block_plan)

    def plaintext_aggregate_delta_for_block(self, block_plan: BlockPlan) -> torch.Tensor | None:
        return self._central_noise_by_key.get((block_plan.recipient_client_id, block_plan.block_id))


class LocalDiagnosticPerturbation(_GaussianPayloadProtection):
    def __init__(
        self,
        clip_norm: float,
        noise_multiplier: float,
        diagnostic_seed: int,
    ) -> None:
        if noise_multiplier < 0:
            raise ValueError("diagnostic noise multiplier must be non-negative.")
        super().__init__(
            clip_norm=clip_norm,
            noise_std=float(clip_norm * noise_multiplier),
            noise_multiplier=noise_multiplier,
            random_seed=diagnostic_seed,
            name="local_diagnostic_perturbation",
            protection_scope="local_source_contribution",
            formal_differential_privacy=False,
            payload_protection_kind="diagnostic_gaussian_perturbation",
            payload_dp_claim=None,
            claim_scope=None,
            privacy_unit=None,
            adjacency=None,
            sensitivity_l2=None,
            epsilon=None,
            delta=None,
            calibration=None,
            trust_boundary=None,
            duplicate_release_policy=None,
            composition=None,
        )

    def protect_compact_contribution_before_blocking(
        self,
        contribution: ClientFeatureContribution,
        metadata: dict[str, Any] | None = None,
    ) -> ClientFeatureContribution:
        clipped = self._clip_rows(contribution.contribution)
        noised = clipped + self._noise_like(clipped)
        return replace(contribution, contribution=noised)


class LocalRowDPPayloadProtection(_GaussianPayloadProtection):
    def __init__(
        self,
        clip_norm: float,
        epsilon: float,
        delta: float,
        payload_dp_seed: int,
        *,
        calibration: str = "classical_gaussian",
    ) -> None:
        if calibration not in {"classical_gaussian", "analytic_gaussian"}:
            raise ValueError(f"Unsupported payload DP calibration: {calibration}")
        sensitivity_l2 = float(2.0 * clip_norm)
        if calibration == "classical_gaussian":
            noise_std = classical_gaussian_sigma(
                epsilon=epsilon,
                delta=delta,
                sensitivity_l2=sensitivity_l2,
            )
        else:
            noise_std = analytic_gaussian_sigma(
                epsilon=epsilon,
                delta=delta,
                sensitivity_l2=sensitivity_l2,
            )
        super().__init__(
            clip_norm=clip_norm,
            noise_std=noise_std,
            noise_multiplier=None,
            random_seed=payload_dp_seed,
            name="local_row_dp",
            requested_name="local_row_dp",
            protection_scope="local_source_contribution",
            formal_differential_privacy=True,
            payload_protection_kind="calibrated_contribution_row_dp",
            payload_dp_claim="contribution_row_substitution_dp",
            claim_scope="feature_pretraining_contribution_payload_only",
            privacy_unit="contribution_row",
            adjacency="substitution",
            sensitivity_l2=sensitivity_l2,
            epsilon=float(epsilon),
            delta=float(delta),
            calibration=calibration,
            trust_boundary=None,
            duplicate_release_policy="reuse_noised_value",
            composition="parallel_with_reused_duplicate_release",
        )
        self._release_counts: Counter[tuple[str, int]] = Counter()

    def protect_compact_contribution_before_blocking(
        self,
        contribution: ClientFeatureContribution,
        metadata: dict[str, Any] | None = None,
    ) -> ClientFeatureContribution:
        clipped = self._clip_rows(contribution.contribution)
        noised = clipped + self._noise_like(clipped)
        return replace(contribution, contribution=noised)

    def record_block_release(self, source_client_id: str, row_ids: torch.Tensor) -> None:
        for row in row_ids.detach().cpu().long().tolist():
            self._release_counts[(str(source_client_id), int(row))] += 1

    def _release_count_metadata(self) -> dict[str, Any]:
        return {
            "num_noised_releases": self._rows_noised,
            "num_unique_private_units": self._rows_noised,
            "max_release_count_per_unit": max(self._release_counts.values()) if self._release_counts else 0,
        }

    def _base_diagnostics(self) -> dict[str, Any]:
        diagnostics = super()._base_diagnostics()
        diagnostics["trust_boundary"] = "local_source"
        return diagnostics


class CentralRowDPPayloadProtection(_GaussianPayloadProtection):
    def __init__(
        self,
        clip_norm: float,
        epsilon: float,
        delta: float,
        payload_dp_seed: int,
        *,
        calibration: str = "classical_gaussian",
    ) -> None:
        if calibration not in {"classical_gaussian", "analytic_gaussian"}:
            raise ValueError(f"Unsupported payload DP calibration: {calibration}")
        sensitivity_l2 = float(2.0 * clip_norm)
        if calibration == "classical_gaussian":
            noise_std = classical_gaussian_sigma(
                epsilon=epsilon,
                delta=delta,
                sensitivity_l2=sensitivity_l2,
            )
        else:
            noise_std = analytic_gaussian_sigma(
                epsilon=epsilon,
                delta=delta,
                sensitivity_l2=sensitivity_l2,
            )
        super().__init__(
            clip_norm=clip_norm,
            noise_std=noise_std,
            noise_multiplier=None,
            random_seed=payload_dp_seed,
            name="central_row_dp",
            requested_name="central_row_dp",
            protection_scope="central_aggregate",
            formal_differential_privacy=True,
            payload_protection_kind="calibrated_contribution_row_dp",
            payload_dp_claim="contribution_row_substitution_dp",
            claim_scope="feature_pretraining_contribution_payload_only",
            privacy_unit="contribution_row",
            adjacency="substitution",
            sensitivity_l2=sensitivity_l2,
            epsilon=float(epsilon),
            delta=float(delta),
            calibration=calibration,
            trust_boundary=None,
            duplicate_release_policy="reuse_noised_value",
            composition="parallel_with_reused_duplicate_release",
        )
        self._central_noise_by_row: dict[int, torch.Tensor] = {}
        self._row_release_counts: Counter[int] = Counter()
        self._private_units: set[tuple[str, int]] = set()

    def protect_compact_contribution_before_blocking(
        self,
        contribution: ClientFeatureContribution,
        metadata: dict[str, Any] | None = None,
    ) -> ClientFeatureContribution:
        client_id = str((metadata or {}).get("client_id", contribution.client_id))
        for row in contribution.target_global_node_ids.detach().cpu().long().tolist():
            self._private_units.add((client_id, int(row)))
        clipped = self._clip_rows(contribution.contribution)
        return replace(contribution, contribution=clipped)

    def _noise_for_row(self, row_id: int, feature_dim: int) -> torch.Tensor:
        if row_id not in self._central_noise_by_row:
            self._central_noise_by_row[row_id] = self._sample_noise((1, feature_dim), dtype=torch.float).view(feature_dim)
        return self._central_noise_by_row[row_id]

    def protect_aggregate_before_delivery(
        self,
        aggregate,
        *,
        transport=None,
        block_plan: BlockPlan | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if transport is None or block_plan is None:
            raise ValueError("central_row_dp aggregate protection requires transport and block_plan.")
        feature_dim = int(block_plan.shape[1])
        noise = torch.zeros(block_plan.shape, dtype=torch.float)
        for idx, row_id in enumerate(block_plan.row_ids.detach().cpu().long().tolist()):
            row_id = int(row_id)
            self._row_release_counts[row_id] += 1
            noise[idx] = self._noise_for_row(row_id, feature_dim)
        return transport.add_plaintext_to_payload_block(aggregate, noise, block_plan)

    def _release_count_metadata(self) -> dict[str, Any]:
        if self._private_units:
            max_release_count = max(self._row_release_counts.get(row, 0) for _, row in self._private_units)
        else:
            max_release_count = max(self._row_release_counts.values()) if self._row_release_counts else 0
        return {
            "num_noised_releases": len(self._central_noise_by_row),
            "num_unique_private_units": len(self._private_units),
            "max_release_count_per_unit": max_release_count,
        }

    def _base_diagnostics(self) -> dict[str, Any]:
        diagnostics = super()._base_diagnostics()
        diagnostics["rows_noised"] = len(self._central_noise_by_row)
        return diagnostics


@dataclass(frozen=True)
class _UnsupportedPayloadProtection(PayloadProtectionPolicy):
    name: str

    def __post_init__(self) -> None:
        raise NotImplementedError(
            f"feature-pretrain payload protection {self.name!r} is planned but not implemented. "
            "Supported policies are 'none', 'central_diagnostic_perturbation', "
            "'local_diagnostic_perturbation', 'central_row_dp', and 'local_row_dp'."
        )


class KMinPayloadFilter(_UnsupportedPayloadProtection):
    def __init__(self) -> None:
        super().__init__("k_filter")


class CentralRowDPKFilter(_UnsupportedPayloadProtection):
    def __init__(self) -> None:
        super().__init__("central_row_dp_k_filter")


def make_payload_protection_policy(
    name: str,
    *,
    clip_norm: float | None = None,
    diagnostic_noise_multiplier: float | None = None,
    diagnostic_seed: int | None = None,
    payload_dp_seed: int | None = None,
    epsilon: float | None = None,
    delta: float | None = None,
    calibration: str = "classical_gaussian",
) -> PayloadProtectionPolicy:
    if name == "none":
        policy = NoPayloadProtection()
        policy.requested_name = name
        return policy
    if name == "k_filter":
        return KMinPayloadFilter()
    if name == "central_row_dp_k_filter":
        return CentralRowDPKFilter()
    if name in DIAGNOSTIC_PERTURBATION_MODES:
        if clip_norm is None or diagnostic_noise_multiplier is None or diagnostic_seed is None:
            raise ValueError(f"{name} requires clip_norm, diagnostic_noise_multiplier, and diagnostic_seed.")
        if name == "central_diagnostic_perturbation":
            return CentralDiagnosticPerturbation(clip_norm, diagnostic_noise_multiplier, diagnostic_seed)
        return LocalDiagnosticPerturbation(clip_norm, diagnostic_noise_multiplier, diagnostic_seed)
    if name in FORMAL_ROW_DP_MODES:
        if clip_norm is None or epsilon is None or delta is None or payload_dp_seed is None:
            raise ValueError(f"{name} requires clip_norm, epsilon, delta, and payload_dp_seed.")
        if name == "central_row_dp":
            return CentralRowDPPayloadProtection(clip_norm, epsilon, delta, payload_dp_seed, calibration=calibration)
        return LocalRowDPPayloadProtection(clip_norm, epsilon, delta, payload_dp_seed, calibration=calibration)
    raise ValueError(f"Unsupported feature-pretrain payload protection: {name}")
