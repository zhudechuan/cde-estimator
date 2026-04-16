"""Constraint builders for portfolio optimisation with CDE.

Provides factory functions to construct the inequality (A, b) and equality
(C, d) constraint matrices commonly used in portfolio selection problems.
Users can combine these or supply their own matrices directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class LinearConstraints:
    """Container for linear inequality and equality constraints.

    Represents the system:
        Aw <= b   (inequality)
        Cw = d    (equality)

    Attributes
    ----------
    A : ndarray of shape (k, p)
        Inequality constraint matrix.
    b : ndarray of shape (k,)
        Inequality constraint RHS.
    C : ndarray of shape (l, p)
        Equality constraint matrix.
    d : ndarray of shape (l,)
        Equality constraint RHS.
    """

    A: NDArray[np.float64]
    b: NDArray[np.float64]
    C: NDArray[np.float64]
    d: NDArray[np.float64]

    @property
    def k(self) -> int:
        """Number of inequality constraints."""
        return self.A.shape[0]

    @property
    def l(self) -> int:
        """Number of equality constraints."""
        return self.C.shape[0]

    @property
    def p(self) -> int:
        """Problem dimension."""
        return self.A.shape[1]

    def append_inequality(
        self, A_new: NDArray[np.float64], b_new: NDArray[np.float64]
    ) -> LinearConstraints:
        """Return a new LinearConstraints with additional inequality rows.

        Parameters
        ----------
        A_new : ndarray of shape (m, p)
            Additional inequality constraint rows.
        b_new : ndarray of shape (m,)
            Corresponding RHS values.

        Returns
        -------
        LinearConstraints
            New instance with the appended constraints.
        """
        return LinearConstraints(
            A=np.vstack([self.A, A_new]),
            b=np.concatenate([self.b, b_new]),
            C=self.C,
            d=self.d,
        )


def budget_constraint(p: int, total: float = 1.0) -> LinearConstraints:
    """Full-investment (budget) equality constraint: sum(w) = total.

    Parameters
    ----------
    p : int
        Number of assets.
    total : float, default=1.0
        Target sum of weights (1.0 for fully invested, 0.0 for
        transaction-cost formulation).

    Returns
    -------
    LinearConstraints
        Constraints with no inequality rows and one equality row.
    """
    C = np.ones((1, p))
    d = np.array([total])
    A = np.empty((0, p))
    b = np.empty(0)
    return LinearConstraints(A=A, b=b, C=C, d=d)


def liquidity_constraint(
    p: int,
    liquid_indices: NDArray[np.intp],
    illiquid_coef: float = 1.0,
    very_illiquid_coef: float = 2.0,
    illiquid_split: float = 0.5,
    allowance_factor: float = 0.1,
) -> LinearConstraints:
    """Liquidity-weighted exposure constraint.

    Assigns zero coefficient to liquid assets, ``illiquid_coef`` to the
    first half of illiquid assets, and ``very_illiquid_coef`` to the rest.
    The total weighted exposure is bounded by
    ``allowance_factor * n_illiquid``.

    Parameters
    ----------
    p : int
        Number of assets.
    liquid_indices : ndarray of int
        Indices of liquid assets (coefficient = 0).
    illiquid_coef : float, default=1.0
        Coefficient for moderately illiquid assets.
    very_illiquid_coef : float, default=2.0
        Coefficient for very illiquid assets.
    illiquid_split : float, default=0.5
        Fraction of non-liquid assets classified as moderately illiquid.
    allowance_factor : float, default=0.1
        Multiplier for the number of illiquid assets to form the RHS bound.

    Returns
    -------
    LinearConstraints
        Single inequality row encoding the liquidity constraint,
        with no equality constraints.
    """
    A = np.zeros((1, p))
    illiquid_mask = np.ones(p, dtype=bool)
    illiquid_mask[liquid_indices] = False
    illiquid_idx = np.where(illiquid_mask)[0]
    n_illiquid = len(illiquid_idx)
    n_moderate = int(n_illiquid * illiquid_split)

    A[0, illiquid_idx[:n_moderate]] = illiquid_coef
    A[0, illiquid_idx[n_moderate:]] = very_illiquid_coef

    b = np.array([allowance_factor * n_illiquid])
    C = np.empty((0, p))
    d = np.empty(0)
    return LinearConstraints(A=A, b=b, C=C, d=d)


def volume_liquidity_constraint(
    p: int,
    volume: NDArray[np.float64],
    n_liquid: int = 10,
    moderate_coef: float = 1.0,
    severe_coef: float = 4.0,
    allowance_factor: float = 0.1,
) -> LinearConstraints:
    """Volume-based liquidity constraint.

    Ranks assets by trading volume and assigns tiered coefficients.

    Parameters
    ----------
    p : int
        Number of assets.
    volume : ndarray of shape (p,)
        Average trading volume per asset.
    n_liquid : int, default=10
        Number of most-liquid assets (coefficient = 0).
    moderate_coef : float, default=1.0
        Coefficient for moderately illiquid assets.
    severe_coef : float, default=4.0
        Coefficient for severely illiquid assets.
    allowance_factor : float, default=0.1
        Multiplier for the number of non-liquid assets.

    Returns
    -------
    LinearConstraints
        Single inequality row, no equality constraints.
    """
    sorted_idx = np.argsort(volume)[::-1]
    remaining = p - n_liquid
    half_remaining = remaining // 2

    A = np.zeros((1, p))
    A[0, sorted_idx[n_liquid : n_liquid + half_remaining]] = moderate_coef
    A[0, sorted_idx[n_liquid + half_remaining :]] = severe_coef
    b = np.array([allowance_factor * remaining])

    C = np.empty((0, p))
    d = np.empty(0)
    return LinearConstraints(A=A, b=b, C=C, d=d)


def gross_exposure_constraint(
    p: int,
    sigma: NDArray[np.float64],
    w0: NDArray[np.float64],
    bound_per_asset: float = 0.0,
    top_variance_fraction: float = 0.5,
) -> LinearConstraints:
    """Per-asset exposure constraint for high-variance assets.

    For the top ``top_variance_fraction`` assets by variance, constrains
    -w_i <= bound_per_asset + w0_i, preventing excessive concentration.

    Parameters
    ----------
    p : int
        Number of assets.
    sigma : ndarray of shape (p, p)
        Covariance matrix.
    w0 : ndarray of shape (p,)
        Previous-period portfolio weights.
    bound_per_asset : float, default=0.0
        Base bound (typically 1/p in the original code).
    top_variance_fraction : float, default=0.5
        Fraction of assets (by highest variance) to constrain.

    Returns
    -------
    LinearConstraints
        Inequality rows for the per-asset bounds, no equality constraints.
    """
    var = np.diag(sigma)
    n_constrained = int(p * top_variance_fraction)
    indices = np.argsort(var)[-n_constrained:]

    e = np.zeros((n_constrained, p))
    e[np.arange(n_constrained), indices] = 1.0

    A = -e
    b = np.ones(n_constrained) * bound_per_asset + e @ w0

    C = np.empty((0, p))
    d = np.empty(0)
    return LinearConstraints(A=A, b=b, C=C, d=d)


def combine_constraints(*constraints: LinearConstraints) -> LinearConstraints:
    """Merge multiple LinearConstraints into one.

    Stacks all inequality rows and all equality rows. The problem
    dimension p must be consistent across all inputs.

    Parameters
    ----------
    *constraints : LinearConstraints
        One or more constraint objects to combine.

    Returns
    -------
    LinearConstraints
        Merged constraints.

    Raises
    ------
    ValueError
        If the constraints have inconsistent dimensions.
    """
    if not constraints:
        raise ValueError("At least one constraint object required")

    ps = set()
    for c in constraints:
        if c.A.shape[1] > 0:
            ps.add(c.A.shape[1])
        if c.C.shape[1] > 0:
            ps.add(c.C.shape[1])
    if len(ps) > 1:
        raise ValueError(f"Inconsistent dimensions: {ps}")

    p = ps.pop() if ps else 0

    A_parts = [c.A for c in constraints if c.A.shape[0] > 0]
    b_parts = [c.b for c in constraints if c.b.shape[0] > 0]
    C_parts = [c.C for c in constraints if c.C.shape[0] > 0]
    d_parts = [c.d for c in constraints if c.d.shape[0] > 0]

    A = np.vstack(A_parts) if A_parts else np.empty((0, p))
    b = np.concatenate(b_parts) if b_parts else np.empty(0)
    C = np.vstack(C_parts) if C_parts else np.empty((0, p))
    d = np.concatenate(d_parts) if d_parts else np.empty(0)

    return LinearConstraints(A=A, b=b, C=C, d=d)
