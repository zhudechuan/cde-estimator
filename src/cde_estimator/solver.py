"""Core CDE solver using CPLEX via docplex.

This module implements the Constrained Dantzig-type Estimator (CDE) with
general linear inequality constraints, solved as a mixed-integer linear
program (MILP) via the KKT complementarity formulation. It also provides
the Self-Calibrated CDE (SC-CDE) extension that introduces an adaptive
scale variable tau, eliminating the need for cross-validation.

Reference
---------
Pun, C. S. & Zhu, D. (2024). Constrained Dantzig-type Estimator with
inequality constraints for high-dimensional sparse learning.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .exceptions import InfeasibleError


def _get_cplex_model():
    """Lazy import of docplex Model to allow package import without CPLEX."""
    try:
        from docplex.mp.model import Model
    except ImportError as exc:
        raise ImportError(
            "docplex is required for the CDE solver. Install it with:\n"
            "  pip install docplex\n"
            "You also need IBM CPLEX installed. Free academic licences are "
            "available at https://www.ibm.com/academic"
        ) from exc
    return Model

logger = logging.getLogger(__name__)


def find_lambda_max(
    sigma: NDArray[np.float64],
    eta: NDArray[np.float64],
    p: int,
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    k: int,
    C: NDArray[np.float64],
    d: NDArray[np.float64],
    l: int,
    factor: float = 1.0,
    big_M: float = 100.0,
) -> Tuple[float, float]:
    """Find the maximum tuning parameter lambda via the two-phase LP/MILP.

    Phase 1 (LP): Compute the minimum l1-norm subject to the linear
    constraints (primal feasibility only, no KKT stationarity).

    Phase 2 (MILP): Minimise lambda such that KKT stationarity is
    satisfied at the l1-minimising solution, using big-M linearisation
    of the complementarity constraints.

    Parameters
    ----------
    sigma : ndarray of shape (p, p)
        Sample covariance matrix.
    eta : ndarray of shape (p,)
        Return signal vector.
    p : int
        Number of assets / variables.
    A : ndarray of shape (k, p)
        Inequality constraint coefficient matrix (Aw <= b).
    b : ndarray of shape (k,)
        Inequality constraint right-hand side.
    k : int
        Number of inequality constraints.
    C : ndarray of shape (l, p)
        Equality constraint coefficient matrix (Cw = d).
    d : ndarray of shape (l,)
        Equality constraint right-hand side.
    l : int
        Number of equality constraints.
    factor : float, default=1.0
        Scaling factor for numerical conditioning of the KKT residual.
    big_M : float, default=100.0
        Big-M constant for complementarity linearisation.

    Returns
    -------
    lambda_max : float
        The maximum lambda at which the KKT system is feasible.
    new_factor : float
        Rescaled factor such that lambda_max maps to ~100 in scaled units.

    Raises
    ------
    InfeasibleError
        If either the LP or MILP is infeasible.
    """
    # ---- Phase 1: minimum l1-norm ----
    Model = _get_cplex_model()
    mdl = Model("cde_lambda_max_lp1")
    mdl.parameters.read.scale = -1

    w = np.array(
        mdl.continuous_var_list([f"w{i}" for i in range(p)], lb=-mdl.infinity)
    )

    mdl.minimize(mdl.sum(mdl.abs(w[i]) for i in range(p)))

    for row in range(k):
        mdl.add_constraint(mdl.dot(w, A[row]) <= b[row])
    for row in range(l):
        mdl.add_constraint(mdl.dot(w, C[row]) == d[row])

    sol = mdl.solve()
    if mdl.solve_status.value != 2:
        mdl.clear()
        raise InfeasibleError("lambda_max_lp1", "minimum l1-norm LP is infeasible")

    lp1_norm = sol.get_objective_value()
    mdl.clear()
    logger.debug("Phase 1 l1-norm: %.6f", lp1_norm)

    # ---- Phase 2: minimise lambda subject to KKT ----
    mdl = Model("cde_lambda_max_lp2")
    mdl.parameters.read.scale = -1

    w = np.array(
        mdl.continuous_var_list([f"w{i}" for i in range(p)], lb=-mdl.infinity)
    )
    gamma_h = np.array(
        mdl.continuous_var_list([f"gamma_h{i}" for i in range(k)], lb=0)
    )
    gamma_g = np.array(
        mdl.continuous_var_list([f"gamma_g{i}" for i in range(l)], lb=-mdl.infinity)
    )
    y = np.array(mdl.binary_var_list([f"y{i}" for i in range(k)]))
    lam = mdl.continuous_var(name="lambda")

    mdl.minimize(lam)

    # KKT stationarity: |factor * (Sigma w - eta + A^T gamma_h + C^T gamma_g)| <= lambda
    for row in range(p):
        kkt_expr = factor * (
            mdl.dot(w, sigma[row])
            - eta[row]
            + mdl.dot(gamma_h, A.T[row])
            + mdl.dot(gamma_g, C.T[row])
        )
        mdl.add_constraint(kkt_expr - lam <= 0)
        mdl.add_constraint(kkt_expr + lam >= 0)

    # Primal feasibility + complementarity (big-M)
    for row in range(k):
        aw = mdl.dot(w, A[row])
        mdl.add_constraint(aw <= b[row])
        mdl.add_constraint(aw - b[row] <= big_M * y[row])
        mdl.add_constraint(b[row] - aw <= big_M * y[row])
        mdl.add_constraint(gamma_h[row] <= big_M * (1 - y[row]))

    for row in range(l):
        mdl.add_constraint(mdl.dot(w, C[row]) == d[row])

    # l1-norm matches Phase 1 optimum
    mdl.add_constraint(mdl.sum(mdl.abs(w[i]) for i in range(p)) == lp1_norm)

    sol = mdl.solve()
    if mdl.solve_status.value != 2:
        mdl.clear()
        raise InfeasibleError("lambda_max_lp2", "KKT MILP is infeasible")

    lambda_max = sol.get_value(lam)
    new_factor = (100.0 / lambda_max * factor) if lambda_max != 0 else factor
    mdl.clear()

    logger.debug("lambda_max: %.6f, new_factor: %.6f", lambda_max, new_factor)
    return lambda_max, new_factor


def solve_cde(
    sigma: NDArray[np.float64],
    eta: NDArray[np.float64],
    p: int,
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    k: int,
    C: NDArray[np.float64],
    d: NDArray[np.float64],
    l: int,
    factor: float,
    lambda_scaled: float,
    big_M: float = 100.0,
    time_limit: Optional[float] = None,
) -> NDArray[np.float64]:
    """Solve the CDE with inequality constraints for a given lambda.

    Minimises ||w||_1 subject to:
        |factor * (Sigma w - eta + A^T gamma_h + C^T gamma_g)| <= lambda  (stationarity)
        Aw <= b, Cw = d                                                    (primal feasibility)
        gamma_h >= 0, complementarity via big-M                            (dual feasibility)

    Parameters
    ----------
    sigma : ndarray of shape (p, p)
        Sample covariance matrix.
    eta : ndarray of shape (p,)
        Return signal vector.
    p : int
        Problem dimension.
    A : ndarray of shape (k, p)
        Inequality constraint matrix.
    b : ndarray of shape (k,)
        Inequality RHS.
    k : int
        Number of inequality constraints.
    C : ndarray of shape (l, p)
        Equality constraint matrix.
    d : ndarray of shape (l,)
        Equality RHS.
    l : int
        Number of equality constraints.
    factor : float
        Scaling factor (typically from ``find_lambda_max``).
    lambda_scaled : float
        Tuning parameter in scaled units.
    big_M : float, default=100.0
        Big-M constant for complementarity.
    time_limit : float, optional
        Solver time limit in seconds.

    Returns
    -------
    w : ndarray of shape (p,)
        Estimated coefficient (portfolio weight) vector.

    Raises
    ------
    InfeasibleError
        If the MILP is infeasible.
    """
    Model = _get_cplex_model()
    mdl = Model(name="cde_core")
    mdl.parameters.read.scale = -1
    if time_limit is not None:
        mdl.set_time_limit(time_limit)

    # Variables
    w = np.array(
        mdl.continuous_var_list([f"w{i}" for i in range(p)], lb=-mdl.infinity)
    )
    gamma_h = np.array(
        mdl.continuous_var_list([f"gamma_h{i}" for i in range(k)], lb=0)
    )
    gamma_g = np.array(
        mdl.continuous_var_list([f"gamma_g{i}" for i in range(l)], lb=-mdl.infinity)
    )
    y = np.array(mdl.binary_var_list([f"y{i}" for i in range(k)]))

    # Objective: minimise l1-norm
    mdl.minimize(mdl.sum(mdl.abs(w[i]) for i in range(p)))

    # KKT stationarity
    for row in range(p):
        kkt_expr = factor * (
            mdl.dot(w, sigma[row])
            - eta[row]
            + mdl.dot(gamma_h, A.T[row])
            + mdl.dot(gamma_g, C.T[row])
        )
        mdl.add_constraint(kkt_expr - lambda_scaled <= 0)
        mdl.add_constraint(kkt_expr + lambda_scaled >= 0)

    # Primal feasibility + complementarity
    for row in range(k):
        aw = mdl.dot(w, A[row])
        mdl.add_constraint(aw <= b[row])
        mdl.add_constraint(aw - b[row] <= big_M * y[row])
        mdl.add_constraint(b[row] - aw <= big_M * y[row])
        mdl.add_constraint(gamma_h[row] <= big_M * (1 - y[row]))

    for row in range(l):
        mdl.add_constraint(mdl.dot(w, C[row]) == d[row])

    sol = mdl.solve()
    if mdl.solve_status.value != 2:
        mdl.clear()
        raise InfeasibleError("cde_core", f"lambda_scaled={lambda_scaled:.4f}")

    result = np.array(sol.get_values(w))
    mdl.clear()
    return result


def find_lambda_max_equality(
    sigma: NDArray[np.float64],
    eta: NDArray[np.float64],
    p: int,
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    k: int,
    factor: float = 1.0,
) -> Tuple[float, float]:
    """Find lambda_max for the equality-only CDE (pure LP).

    When the CDE has only equality constraints (no inequality), no big-M
    complementarity is needed, and the KKT system is a linear program.
    This is substantially faster than the MILP formulation.

    Parameters
    ----------
    sigma : ndarray of shape (p, p)
        Generalised covariance matrix (e.g., Sigma_tilde for precision
        matrix estimation).
    eta : ndarray of shape (p,)
        Signal vector (e.g., vec(I_p) for precision matrix estimation).
    p : int
        Problem dimension.
    A : ndarray of shape (k, p)
        Equality constraint matrix (Aw = b).
    b : ndarray of shape (k,)
        Equality constraint RHS.
    k : int
        Number of equality constraints.
    factor : float, default=1.0
        Scaling factor for numerical conditioning.

    Returns
    -------
    lambda_max : float
        Maximum lambda at which the KKT system is feasible.
    new_factor : float
        Rescaled factor.

    Raises
    ------
    InfeasibleError
        If either LP is infeasible.
    """
    Model = _get_cplex_model()

    # ---- Phase 1: minimum l1-norm ----
    mdl = Model("cde_eq_lambda_max_lp1")
    mdl.parameters.read.scale = -1
    mdl.parameters.lpmethod = 4

    w = np.array(
        mdl.continuous_var_list([f"w{i}" for i in range(p)], lb=-mdl.infinity)
    )
    mdl.minimize(mdl.sum(mdl.abs(w[i]) for i in range(p)))

    for row in range(k):
        mdl.add_constraint(mdl.dot(w, A[row]) == b[row])

    sol = mdl.solve()
    if mdl.solve_status.value != 2:
        mdl.clear()
        raise InfeasibleError("lambda_max_eq_lp1", "minimum l1-norm LP is infeasible")

    lp1_norm = sol.get_objective_value()
    mdl.clear()

    # ---- Phase 2: minimise lambda subject to KKT stationarity ----
    mdl = Model("cde_eq_lambda_max_lp2")
    mdl.parameters.read.scale = -1
    mdl.parameters.lpmethod = 4

    w = np.array(
        mdl.continuous_var_list([f"w{i}" for i in range(p)], lb=-mdl.infinity)
    )
    gamma = np.array(
        mdl.continuous_var_list([f"gamma{i}" for i in range(k)], lb=-mdl.infinity)
    )
    lam = mdl.continuous_var(name="lambda")

    mdl.minimize(lam)

    for row in range(p):
        kkt_expr = factor * (
            mdl.dot(w, sigma[row])
            - eta[row]
            + mdl.dot(gamma, A.T[row])
        )
        mdl.add_constraint(kkt_expr - lam <= 0)
        mdl.add_constraint(kkt_expr + lam >= 0)

    for row in range(k):
        mdl.add_constraint(mdl.dot(w, A[row]) == b[row])

    mdl.add_constraint(mdl.sum(mdl.abs(w[i]) for i in range(p)) == lp1_norm)

    sol = mdl.solve()
    if mdl.solve_status.value != 2:
        mdl.clear()
        raise InfeasibleError("lambda_max_eq_lp2", "KKT LP is infeasible")

    lambda_max = sol.get_value(lam)
    new_factor = (1.0 / lambda_max * factor) if lambda_max != 0 else factor
    mdl.clear()

    return lambda_max, new_factor


def solve_cde_equality(
    sigma: NDArray[np.float64],
    eta: NDArray[np.float64],
    p: int,
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    k: int,
    factor: float,
    lambda_scaled: float,
    time_limit: Optional[float] = None,
) -> NDArray[np.float64]:
    """Solve the equality-only CDE as a linear program.

    When there are no inequality constraints, the CDE reduces to a pure LP
    with KKT stationarity and equality constraints. This avoids binary
    variables entirely and is much faster for applications like precision
    matrix estimation.

    Parameters
    ----------
    sigma : ndarray of shape (p, p)
        Generalised covariance matrix.
    eta : ndarray of shape (p,)
        Signal vector.
    p : int
        Problem dimension.
    A : ndarray of shape (k, p)
        Equality constraint matrix.
    b : ndarray of shape (k,)
        Equality constraint RHS.
    k : int
        Number of equality constraints.
    factor : float
        Scaling factor (from ``find_lambda_max_equality``).
    lambda_scaled : float
        Tuning parameter.
    time_limit : float, optional
        Solver time limit in seconds.

    Returns
    -------
    w : ndarray of shape (p,)
        Estimated coefficient vector.

    Raises
    ------
    InfeasibleError
        If the LP is infeasible.
    """
    Model = _get_cplex_model()
    mdl = Model(name="cde_eq_core")
    mdl.parameters.read.scale = -1
    mdl.parameters.lpmethod = 4
    if time_limit is not None:
        mdl.set_time_limit(time_limit)

    w = np.array(
        mdl.continuous_var_list([f"w{i}" for i in range(p)], lb=-mdl.infinity)
    )
    gamma = np.array(
        mdl.continuous_var_list([f"gamma{i}" for i in range(k)], lb=-mdl.infinity)
    )

    mdl.minimize(mdl.sum(mdl.abs(w[i]) for i in range(p)))

    # KKT stationarity
    for row in range(p):
        kkt_expr = factor * (
            mdl.dot(w, sigma[row])
            - eta[row]
            + mdl.dot(gamma, A.T[row])
        )
        mdl.add_constraint(kkt_expr - lambda_scaled <= 0)
        mdl.add_constraint(kkt_expr + lambda_scaled >= 0)

    # Equality constraints
    for row in range(k):
        mdl.add_constraint(mdl.dot(w, A[row]) == b[row])

    sol = mdl.solve()
    if mdl.solve_status.value != 2:
        mdl.clear()
        raise InfeasibleError("cde_eq_core", f"lambda_scaled={lambda_scaled:.4f}")

    result = np.array(sol.get_values(w))
    mdl.clear()
    return result


def solve_self_calibrated_cde(
    sigma: NDArray[np.float64],
    eta: NDArray[np.float64],
    p: int,
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    k: int,
    C: NDArray[np.float64],
    d: NDArray[np.float64],
    l: int,
    factor: float,
    lambda_scaled: float,
    c_const: float = 3.0,
    big_M: float = 100.0,
    time_limit: Optional[float] = None,
) -> Tuple[NDArray[np.float64], float]:
    """Solve the Self-Calibrated CDE with an adaptive scale variable.

    The SC-CDE jointly estimates the coefficient vector beta and a scale
    variable tau, making the tuning parameter lambda adaptive. This
    eliminates the need for cross-validation by replacing the fixed
    constraint with a penalised objective.

    Formulation::

        min  ||beta||_1 + c * tau
        s.t. |factor * (Sigma beta - eta + A' gamma_h + C' gamma_g)| <= lam * tau
             ||beta||_1 <= tau
             tau >= 0
             A beta <= b,  C beta = d
             gamma_h >= 0
             (A beta - b)' gamma_h = 0  (complementary slackness, via big-M)

    Uses the same factor rescaling as the standard CDE solver. The
    stationarity constraint in the MILP is::

        |factor * (Sigma beta - eta + A' gamma + C' mu)| <= lam_scaled * tau

    which is equivalent to raw-units formulation with lam_raw = lam_scaled / factor.

    Parameters
    ----------
    sigma : ndarray of shape (p, p)
        Sample covariance matrix.
    eta : ndarray of shape (p,)
        Sample mean return vector.
    p : int
        Problem dimension.
    A : ndarray of shape (k, p)
        Inequality constraint matrix (A beta <= b).
    b : ndarray of shape (k,)
        Inequality RHS.
    k : int
        Number of inequality constraints.
    C : ndarray of shape (l, p)
        Equality constraint matrix (C beta = d).
    d : ndarray of shape (l,)
        Equality RHS.
    l : int
        Number of equality constraints.
    factor : float
        Rescaling factor from ``find_lambda_max`` (maps lambda_max to 100).
    lambda_scaled : float
        Lambda in scaled units (same space as CDE's [60, 100] grid).
    c_const : float, default=3.0
        Penalty constant on the scale variable tau.
    big_M : float, default=100.0
        Big-M constant for complementarity linearisation.
    time_limit : float, optional
        Solver time limit in seconds.

    Returns
    -------
    beta : ndarray of shape (p,)
        Estimated coefficient vector.
    tau : float
        Estimated scale variable.

    Raises
    ------
    InfeasibleError
        If the MILP is infeasible.
    """
    Model = _get_cplex_model()
    mdl = Model(name="sc_cde")
    mdl.parameters.read.scale = -1
    if time_limit is not None:
        mdl.set_time_limit(time_limit)

    # Decompose w = w_plus - w_minus for explicit l1 linearisation.
    w_plus = np.array(
        mdl.continuous_var_list([f"wp{i}" for i in range(p)], lb=0)
    )
    w_minus = np.array(
        mdl.continuous_var_list([f"wm{i}" for i in range(p)], lb=0)
    )
    gamma_h = np.array(
        mdl.continuous_var_list([f"gh{i}" for i in range(k)], lb=0)
    )
    gamma_g = np.array(
        mdl.continuous_var_list([f"gg{i}" for i in range(l)], lb=-mdl.infinity)
    )
    y_bin = np.array(mdl.binary_var_list([f"y{i}" for i in range(k)]))
    tau = mdl.continuous_var(name="tau", lb=0)

    # l1 norm expression
    l1_norm = mdl.sum(w_plus[i] + w_minus[i] for i in range(p))

    # Objective: ||beta||_1 + c * tau
    mdl.minimize(l1_norm + c_const * tau)

    # Stationarity: |factor * (Sigma beta - eta + A' gamma_h + C' gamma_g)| <= lam * tau
    for row in range(p):
        kkt = factor * (
            mdl.dot(w_plus - w_minus, sigma[row]) - eta[row]
            + mdl.dot(gamma_h, A.T[row])
            + mdl.dot(gamma_g, C.T[row])
        )
        mdl.add_constraint(kkt - lambda_scaled * tau <= 0)
        mdl.add_constraint(kkt + lambda_scaled * tau >= 0)

    # Scale constraint: ||beta||_1 <= tau
    mdl.add_constraint(l1_norm <= tau)

    # Primal feasibility + complementarity
    for row in range(k):
        aw = mdl.dot(w_plus - w_minus, A[row])
        mdl.add_constraint(aw <= b[row])
        mdl.add_constraint(aw - b[row] <= big_M * y_bin[row])
        mdl.add_constraint(b[row] - aw <= big_M * y_bin[row])
        mdl.add_constraint(gamma_h[row] <= big_M * (1 - y_bin[row]))

    for row in range(l):
        mdl.add_constraint(mdl.dot(w_plus - w_minus, C[row]) == d[row])

    # Solve
    sol = mdl.solve()
    if mdl.solve_status.value != 2:
        mdl.clear()
        raise InfeasibleError(
            "sc_cde", f"lambda_scaled={lambda_scaled:.4f}, c={c_const:.1f}"
        )

    beta = np.array(sol.get_values(w_plus)) - np.array(sol.get_values(w_minus))
    tau_val = sol.get_value(tau)
    mdl.clear()
    return beta, tau_val
