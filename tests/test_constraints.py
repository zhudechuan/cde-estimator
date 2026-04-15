"""Tests for the constraints module (no CPLEX needed)."""

import numpy as np
import pytest

from cde_estimator.constraints import (
    LinearConstraints,
    budget_constraint,
    combine_constraints,
    gross_exposure_constraint,
    liquidity_constraint,
    volume_liquidity_constraint,
)


class TestBudgetConstraint:
    def test_shape(self):
        c = budget_constraint(50, total=1.0)
        assert c.C.shape == (1, 50)
        assert c.d.shape == (1,)
        assert c.A.shape[0] == 0  # no inequality rows
        assert c.k == 0
        assert c.l == 1
        assert c.p == 50

    def test_values(self):
        c = budget_constraint(10, total=0.0)
        np.testing.assert_array_equal(c.C, np.ones((1, 10)))
        np.testing.assert_array_equal(c.d, [0.0])


class TestLiquidityConstraint:
    def test_liquid_assets_have_zero_coef(self):
        p = 20
        liquid = np.array([0, 1, 2])
        c = liquidity_constraint(p, liquid)
        assert c.A.shape == (1, p)
        assert c.A[0, 0] == 0.0
        assert c.A[0, 1] == 0.0
        assert c.A[0, 2] == 0.0

    def test_illiquid_assets_positive(self):
        p = 20
        liquid = np.array([0, 1])
        c = liquidity_constraint(p, liquid)
        illiquid_coefs = c.A[0, 2:]
        assert np.all(illiquid_coefs > 0)


class TestVolumeLiquidity:
    def test_highest_volume_zero_coef(self):
        p = 30
        vol = np.arange(p, dtype=float)  # asset 29 has highest volume
        c = volume_liquidity_constraint(p, vol, n_liquid=5)
        # Top 5 by volume: indices 25-29
        for i in range(25, 30):
            assert c.A[0, i] == 0.0


class TestGrossExposure:
    def test_shape(self):
        p = 20
        sigma = np.eye(p) * np.arange(1, p + 1)
        w0 = np.ones(p) / p
        c = gross_exposure_constraint(p, sigma, w0, bound_per_asset=1 / p)
        # top 50% variance = 10 assets constrained
        assert c.A.shape == (10, p)
        assert c.b.shape == (10,)


class TestCombineConstraints:
    def test_merges_correctly(self):
        p = 20
        c1 = budget_constraint(p)
        c2 = liquidity_constraint(p, np.arange(5))
        merged = combine_constraints(c1, c2)
        assert merged.k == 1  # one liquidity inequality
        assert merged.l == 1  # one budget equality
        assert merged.p == p

    def test_dimension_mismatch_raises(self):
        c1 = budget_constraint(10)
        c2 = budget_constraint(20)
        with pytest.raises(ValueError, match="Inconsistent"):
            combine_constraints(c1, c2)


class TestLinearConstraintsAppend:
    def test_append(self):
        p = 10
        c = budget_constraint(p)
        # Add a dummy inequality
        A_new = np.ones((1, p))
        b_new = np.array([5.0])
        c2 = c.append_inequality(A_new, b_new)
        assert c2.k == 1
        assert c2.l == 1  # equality preserved
