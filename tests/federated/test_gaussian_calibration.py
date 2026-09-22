import math

import pytest

from federated_gcn_aml.federated.payload_protection import (
    CentralRowDPPayloadProtection,
    LocalRowDPPayloadProtection,
    analytic_gaussian_sigma,
)


def _gaussian_delta(epsilon, sigma, sensitivity):
    # Balle-Wang Theorem 8, independent of the production B+/B- root mapping.
    ratio = sensitivity / (2.0 * sigma)
    shift = epsilon * sigma / sensitivity
    cdf = lambda value: 0.5 * math.erfc(-value / math.sqrt(2.0))
    return cdf(ratio - shift) - math.exp(epsilon) * cdf(-ratio - shift)


# Independently solved Theorem 8 with 80-decimal mpmath arithmetic and 300
# bisection steps directly in sigma, not the production alpha parameterization.
PANEL_SIGMAS = [
    (1, 37.306316348159418),
    (2, 19.938124456435367),
    (4, 10.811618495202392),
    (8, 6.002290721989516),
    (16, 3.441774284963935),
    (32, 2.049349002588112),
    (64, 1.267139022470182),
]


@pytest.mark.parametrize("epsilon,expected_sigma", PANEL_SIGMAS)
def test_analytic_panel_matches_independent_reference(epsilon, expected_sigma):
    sigma = analytic_gaussian_sigma(epsilon=epsilon, delta=1e-5, sensitivity_l2=10)
    assert sigma == pytest.approx(expected_sigma, rel=2e-11)
    assert _gaussian_delta(epsilon, sigma, 10) <= 1e-5 * (1 + 1e-10)
    assert _gaussian_delta(epsilon, sigma * (1 - 1e-6), 10) > 1e-5


@pytest.mark.parametrize("policy_class", [LocalRowDPPayloadProtection, CentralRowDPPayloadProtection])
@pytest.mark.parametrize("epsilon,expected_sigma", PANEL_SIGMAS)
def test_formal_policies_use_corrected_panel_calibration(policy_class, epsilon, expected_sigma):
    policy = policy_class(
        clip_norm=5, epsilon=epsilon, delta=1e-5,
        payload_dp_seed=1000045, calibration="analytic_gaussian",
    )
    diagnostics = policy.diagnostics()
    assert diagnostics["payload_dp_noise_std"] == pytest.approx(expected_sigma, rel=2e-11)
    assert diagnostics["payload_dp_sensitivity_l2"] == 10


@pytest.mark.parametrize("epsilon,delta", [(0.1, 0.8), (1, 0.8), (1, 1e-12), (64, 1e-12)])
def test_both_branches_and_small_delta_satisfy_privacy_condition(epsilon, delta):
    sigma = analytic_gaussian_sigma(epsilon=epsilon, delta=delta, sensitivity_l2=2)
    assert _gaussian_delta(epsilon, sigma, 2) == pytest.approx(delta, rel=1e-9)


def test_branch_boundary():
    epsilon = 1.0
    delta = 0.5 - math.exp(epsilon) * 0.5 * math.erfc(math.sqrt(epsilon))
    sigma = analytic_gaussian_sigma(epsilon=epsilon, delta=delta, sensitivity_l2=2)
    assert sigma == pytest.approx(2 / math.sqrt(2 * epsilon), rel=1e-6)
    assert _gaussian_delta(epsilon, sigma, 2) <= delta * (1 + 1e-12)


@pytest.mark.parametrize("tol", [0, -1, math.nan, math.inf])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tolerance"):
        analytic_gaussian_sigma(epsilon=1, delta=1e-5, sensitivity_l2=2, tol=tol)


def test_nonconvergence_fails_closed():
    with pytest.raises(ValueError, match="converge|bracket"):
        analytic_gaussian_sigma(epsilon=1, delta=1e-5, sensitivity_l2=2, max_iterations=1)
