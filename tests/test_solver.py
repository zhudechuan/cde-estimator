"""Tests for the CDE solver (require CPLEX).

These tests use small synthetic problems. They are marked with
``pytest.mark.cplex`` so they can be skipped when CPLEX is unavailable:

    pytest -m "not cplex"      # skip solver tests
    pytest                     # run everything (needs CPLEX)
"""

import numpy as np
import pytest

try:
    from docplex.mp.model import Model as _  # noqa: F401

    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False

cplex = pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX not installed")


@cplex
class TestFindLambdaMax:
    """Test the two-phase lambda_max finder on a tiny problem."""

    @staticmethod
    def _small_problem():
        """p=5 with budget + one inequality."""
        rng = np.random.default_rng(0)
        p = 5
        X = rng.standard_normal((50, p))
        sigma = X.T @ X / 50
        eta = rng.standard_normal(p) * 0.01

        A = np.ones((1, p))  # sum(w) <= 2
        b = np.array([2.0])
        C = np.ones((1, p))  # sum(w) = 1
        d = np.array([1.0])
        k, l = 1, 1
        return sigma, eta, p, A, b, k, C, d, l

    def test_returns_positive_lambda(self):
        from cde_estimator.solver import find_lambda_max

        args = self._small_problem()
        lam_max, factor = find_lambda_max(*args)
        assert lam_max >= 0
        assert factor > 0

    def test_factor_scales_to_100(self):
        from cde_estimator.solver import find_lambda_max

        args = self._small_problem()
        lam_max, factor = find_lambda_max(*args)
        if lam_max > 0:
            assert abs(100.0 / lam_max * 1.0 - factor) < 1e-6


@cplex
class TestSolveCDE:
    """Test the core CDE solver."""

    def test_budget_respected(self):
        from cde_estimator.solver import find_lambda_max, solve_cde

        rng = np.random.default_rng(42)
        p = 8
        X = rng.standard_normal((100, p))
        sigma = X.T @ X / 100
        eta = rng.standard_normal(p) * 0.01

        A = np.ones((1, p)) * 2  # 2*sum(w) <= 3
        b = np.array([3.0])
        C = np.ones((1, p))
        d = np.array([1.0])
        k, l = 1, 1

        lam_max, factor = find_lambda_max(sigma, eta, p, A, b, k, C, d, l)
        w = solve_cde(sigma, eta, p, A, b, k, C, d, l, factor, lambda_scaled=90.0)

        # Budget constraint: sum = 1
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-4)
        # Inequality: 2*sum <= 3
        assert 2 * w.sum() <= 3.0 + 1e-4


@cplex
class TestCDEEstimator:
    """Integration test for the high-level CDEEstimator."""

    def test_fit_returns_result(self):
        from cde_estimator import CDEEstimator
        from cde_estimator.constraints import budget_constraint

        rng = np.random.default_rng(99)
        p = 6
        X = rng.standard_normal((80, p))
        sigma = X.T @ X / 80
        eta = X.mean(axis=0)

        constraints = budget_constraint(p, total=1.0)
        # Need at least one inequality for the MILP
        A_dummy = np.ones((1, p)) * 10
        b_dummy = np.array([100.0])
        constraints = constraints.append_inequality(A_dummy, b_dummy)

        est = CDEEstimator()
        result = est.fit(sigma, eta, constraints, lambda_value=90.0)

        assert result.weights.shape == (p,)
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-4)
        assert result.lambda_selected == 90.0


@cplex
class TestSolveSelfCalibratedCDE:
    """Test the SC-CDE solver."""

    def test_returns_weights_and_tau(self):
        from cde_estimator.solver import find_lambda_max, solve_self_calibrated_cde

        rng = np.random.default_rng(42)
        p = 8
        X = rng.standard_normal((100, p))
        sigma = X.T @ X / 100
        eta = rng.standard_normal(p) * 0.01

        A = np.ones((1, p)) * 2  # 2*sum(w) <= 3
        b = np.array([3.0])
        C = np.ones((1, p))  # sum(w) = 1
        d = np.array([1.0])
        k, l = 1, 1

        lam_max, factor = find_lambda_max(sigma, eta, p, A, b, k, C, d, l)
        w, tau = solve_self_calibrated_cde(
            sigma, eta, p, A, b, k, C, d, l,
            factor, lambda_scaled=90.0, c_const=3.0,
        )

        assert w.shape == (p,)
        assert tau >= 0
        # Budget constraint: sum = 1
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-4)
        # Inequality: 2*sum <= 3
        assert 2 * w.sum() <= 3.0 + 1e-4
        # l1-norm <= tau
        assert np.abs(w).sum() <= tau + 1e-4

    def test_tau_scales_with_c(self):
        """Larger c_const should penalise tau more, yielding smaller tau."""
        from cde_estimator.solver import find_lambda_max, solve_self_calibrated_cde

        rng = np.random.default_rng(7)
        p = 6
        X = rng.standard_normal((80, p))
        sigma = X.T @ X / 80
        eta = rng.standard_normal(p) * 0.01

        A = np.ones((1, p)) * 2
        b = np.array([3.0])
        C = np.ones((1, p))
        d = np.array([1.0])
        k, l = 1, 1

        lam_max, factor = find_lambda_max(sigma, eta, p, A, b, k, C, d, l)

        _, tau_small_c = solve_self_calibrated_cde(
            sigma, eta, p, A, b, k, C, d, l,
            factor, lambda_scaled=90.0, c_const=1.0,
        )
        _, tau_large_c = solve_self_calibrated_cde(
            sigma, eta, p, A, b, k, C, d, l,
            factor, lambda_scaled=90.0, c_const=10.0,
        )

        assert tau_large_c <= tau_small_c + 1e-6


@cplex
class TestSCCDEEstimator:
    """Integration test for the high-level SCCDEEstimator."""

    def test_fit_returns_result(self):
        from cde_estimator import SCCDEEstimator
        from cde_estimator.constraints import budget_constraint

        rng = np.random.default_rng(99)
        p = 6
        X = rng.standard_normal((80, p))
        sigma = X.T @ X / 80
        eta = X.mean(axis=0)

        constraints = budget_constraint(p, total=1.0)
        # Need at least one inequality for the MILP
        A_dummy = np.ones((1, p)) * 10
        b_dummy = np.array([100.0])
        constraints = constraints.append_inequality(A_dummy, b_dummy)

        est = SCCDEEstimator()
        result = est.fit(sigma, eta, constraints, lambda_value=90.0, c_const=3.0)

        assert result.weights.shape == (p,)
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-4)
        assert result.tau >= 0
        assert result.lambda_selected == 90.0
        assert result.c_selected == 3.0
