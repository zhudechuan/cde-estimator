"""CDE Estimator: Constrained Dantzig-type Estimator for high-dimensional sparse learning.

A Python library implementing the Constrained Dantzig-type Estimator (CDE)
with general linear inequality and equality constraints, solved via
mixed-integer linear programming (MILP) or pure LP through IBM CPLEX.

Modules
-------
- ``cde_estimator``: Portfolio selection with inequality constraints (MILP),
  including the Self-Calibrated CDE (SC-CDE) with adaptive scale variable.
- ``cde_estimator.precision``: Precision matrix estimation with equality
  constraints (LP) — see :class:`~cde_estimator.precision.PrecisionMatrixEstimator`.

Example
-------
>>> from cde_estimator import CDEEstimator
>>> from cde_estimator.constraints import budget_constraint, liquidity_constraint, combine_constraints
>>>
>>> estimator = CDEEstimator()
>>> constraints = combine_constraints(
...     budget_constraint(p=100),
...     liquidity_constraint(p=100, liquid_indices=top_10),
... )
>>> result = estimator.fit_cv(sigma, eta, constraints, returns)
"""

from .constraints import (
    LinearConstraints,
    budget_constraint,
    combine_constraints,
    gross_exposure_constraint,
    liquidity_constraint,
    volume_liquidity_constraint,
)
from .estimator import CDEEstimator, CDEResult, SCCDEEstimator, SCCDEResult
from .exceptions import CDEError, InfeasibleError, InputValidationError, SolverError
from .solver import (
    find_lambda_max,
    find_lambda_max_equality,
    solve_cde,
    solve_cde_equality,
    solve_self_calibrated_cde,
)
from .utils import perturb_covariance, validate_dimensions

__version__ = "0.2.0"

__all__ = [
    # Estimator (standard CDE)
    "CDEEstimator",
    "CDEResult",
    # Estimator (self-calibrated CDE)
    "SCCDEEstimator",
    "SCCDEResult",
    # Solver (inequality — MILP)
    "find_lambda_max",
    "solve_cde",
    "solve_self_calibrated_cde",
    # Solver (equality — LP)
    "find_lambda_max_equality",
    "solve_cde_equality",
    # Constraints
    "LinearConstraints",
    "budget_constraint",
    "liquidity_constraint",
    "volume_liquidity_constraint",
    "gross_exposure_constraint",
    "combine_constraints",
    # Utilities
    "perturb_covariance",
    "validate_dimensions",
    # Exceptions
    "CDEError",
    "InfeasibleError",
    "InputValidationError",
    "SolverError",
]
