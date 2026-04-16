"""High-level CDE estimators with cross-validation and grid-search support.

Provides :class:`CDEEstimator` for the standard CDE with K-fold CV, and
:class:`SCCDEEstimator` for the Self-Calibrated CDE with grid search
over (lambda, c) pairs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from .constraints import LinearConstraints
from .exceptions import InfeasibleError
from .solver import find_lambda_max, solve_cde, solve_self_calibrated_cde

logger = logging.getLogger(__name__)


@dataclass
class CDEResult:
    """Container for a single CDE fit result.

    Attributes
    ----------
    weights : ndarray of shape (p,)
        Estimated portfolio weights (or transaction vector).
    lambda_selected : float
        The lambda value used (after CV selection or manual).
    lambda_max : float
        The maximum feasible lambda.
    factor : float
        The scaling factor used in the MILP.
    cv_scores : ndarray or None
        Cross-validation scores if CV was used, else None.
    """

    weights: NDArray[np.float64]
    lambda_selected: float
    lambda_max: float
    factor: float
    cv_scores: Optional[NDArray[np.float64]] = None


class CDEEstimator:
    """Constrained Dantzig-type Estimator for portfolio selection.

    Solves the CDE with general linear inequality and equality constraints
    using the MILP formulation via CPLEX. Supports automatic lambda
    selection through K-fold cross-validation.

    Parameters
    ----------
    lambda_grid : sequence of float, optional
        Grid of candidate lambda values (in scaled units, where lambda_max
        maps to ~100). If None, a default grid is used.
    n_splits : int, default=5
        Number of folds for cross-validation.
    cv_criterion : str, default="variance"
        Cross-validation criterion. Currently supports "variance"
        (minimise portfolio return variance).
    big_M : float, default=100.0
        Big-M constant for complementarity linearisation.
    time_limit : float, optional
        Per-solve time limit in seconds.
    random_state : int, optional
        Random seed for CV fold generation.
    verbose : bool, default=False
        If True, enable debug logging.

    Examples
    --------
    >>> import numpy as np
    >>> from cde_estimator import CDEEstimator
    >>> from cde_estimator.constraints import budget_constraint, liquidity_constraint, combine_constraints
    >>>
    >>> # Setup
    >>> returns = ...  # (n, p) array of asset returns
    >>> sigma = np.cov(returns, rowvar=False, ddof=1)
    >>> eta = returns.mean(axis=0)
    >>> p = sigma.shape[0]
    >>>
    >>> # Build constraints
    >>> budget = budget_constraint(p, total=1.0)
    >>> liquidity = liquidity_constraint(p, liquid_indices=np.arange(10))
    >>> constraints = combine_constraints(budget, liquidity)
    >>>
    >>> # Fit with cross-validation
    >>> estimator = CDEEstimator()
    >>> result = estimator.fit_cv(sigma, eta, constraints, returns)
    >>> print(result.weights)
    """

    def __init__(
        self,
        lambda_grid: Optional[Sequence[float]] = None,
        n_splits: int = 5,
        cv_criterion: str = "variance",
        big_M: float = 100.0,
        time_limit: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        if lambda_grid is None:
            # Default: 100 down to 60 in steps of ~4.5 (18 candidates)
            self.lambda_grid = [i / 40 for i in range(4000, 2400, -90)]
        else:
            self.lambda_grid = list(lambda_grid)

        self.n_splits = n_splits
        self.cv_criterion = cv_criterion
        self.big_M = big_M
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    def fit(
        self,
        sigma: NDArray[np.float64],
        eta: NDArray[np.float64],
        constraints: LinearConstraints,
        lambda_value: float,
    ) -> CDEResult:
        """Fit the CDE for a single specified lambda.

        Parameters
        ----------
        sigma : ndarray of shape (p, p)
            Covariance matrix.
        eta : ndarray of shape (p,)
            Return signal.
        constraints : LinearConstraints
            The inequality and equality constraints.
        lambda_value : float
            Tuning parameter in scaled units.

        Returns
        -------
        CDEResult
            The fitted result.
        """
        p = constraints.p
        A, b, C, d = constraints.A, constraints.b, constraints.C, constraints.d
        k, l = constraints.k, constraints.l

        lambda_max, factor = find_lambda_max(
            sigma, eta, p, A, b, k, C, d, l, factor=1.0, big_M=self.big_M
        )

        w = solve_cde(
            sigma, eta, p, A, b, k, C, d, l,
            factor=factor,
            lambda_scaled=lambda_value,
            big_M=self.big_M,
            time_limit=self.time_limit,
        )

        return CDEResult(
            weights=w,
            lambda_selected=lambda_value,
            lambda_max=lambda_max,
            factor=factor,
        )

    def fit_cv(
        self,
        sigma: NDArray[np.float64],
        eta: NDArray[np.float64],
        constraints: LinearConstraints,
        returns_data: NDArray[np.float64],
        lambda_grid: Optional[Sequence[float]] = None,
    ) -> CDEResult:
        """Fit the CDE with K-fold cross-validation for lambda selection.

        Uses the full covariance and constraints for the solve, but
        evaluates out-of-sample portfolio return variance across CV folds
        to select the best lambda.

        Parameters
        ----------
        sigma : ndarray of shape (p, p)
            Full-sample covariance matrix.
        eta : ndarray of shape (p,)
            Return signal.
        constraints : LinearConstraints
            Linear constraints.
        returns_data : ndarray of shape (n, p)
            Raw return data used for CV fold splitting and evaluation.
        lambda_grid : sequence of float, optional
            Override the instance lambda grid for this call.

        Returns
        -------
        CDEResult
            Result fitted with the best lambda.
        """
        grid = list(lambda_grid) if lambda_grid is not None else self.lambda_grid
        p = constraints.p
        A, b, C, d = constraints.A, constraints.b, constraints.C, constraints.d
        k, l = constraints.k, constraints.l

        # Compute lambda_max and factor on full data
        lam_max, factor = find_lambda_max(
            sigma, eta, p, A, b, k, C, d, l, factor=1.0, big_M=self.big_M
        )

        # Set up CV folds
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        folds = list(kf.split(returns_data))

        cv_scores = np.zeros(len(grid))

        for idx, lam in enumerate(grid):
            fold_returns: List[NDArray] = []

            for train_idx, test_idx in folds:
                try:
                    w = solve_cde(
                        sigma, eta, p, A, b, k, C, d, l,
                        factor=factor,
                        lambda_scaled=lam,
                        big_M=self.big_M,
                        time_limit=self.time_limit,
                    )
                    test_ret = returns_data[test_idx] @ w
                    fold_returns.append(test_ret)
                except InfeasibleError:
                    logger.warning("Lambda %.4f infeasible on a fold, skipping", lam)
                    fold_returns.append(np.array([np.inf]))

            all_returns = np.concatenate(fold_returns)
            if self.cv_criterion == "variance":
                cv_scores[idx] = np.var(all_returns)
            else:
                raise ValueError(f"Unknown cv_criterion: {self.cv_criterion}")

            logger.debug("Lambda %.4f -> CV score %.6e", lam, cv_scores[idx])

        best_idx = np.argmin(cv_scores)
        best_lambda = grid[best_idx]
        logger.info("Best lambda: %.4f (CV score: %.6e)", best_lambda, cv_scores[best_idx])

        # Final fit on full data
        w = solve_cde(
            sigma, eta, p, A, b, k, C, d, l,
            factor=factor,
            lambda_scaled=best_lambda,
            big_M=self.big_M,
            time_limit=self.time_limit,
        )

        return CDEResult(
            weights=w,
            lambda_selected=best_lambda,
            lambda_max=lam_max,
            factor=factor,
            cv_scores=cv_scores,
        )


@dataclass
class SCCDEResult:
    """Container for a Self-Calibrated CDE fit result.

    Attributes
    ----------
    weights : ndarray of shape (p,)
        Estimated portfolio weights (or coefficient vector).
    tau : float
        Estimated scale variable.
    lambda_selected : float
        The lambda value used (in scaled units).
    c_selected : float
        The penalty constant c used.
    lambda_max : float
        The maximum feasible lambda.
    factor : float
        The scaling factor used in the MILP.
    grid_scores : ndarray or None
        Grid search scores if grid search was used, else None.
    """

    weights: NDArray[np.float64]
    tau: float
    lambda_selected: float
    c_selected: float
    lambda_max: float
    factor: float
    grid_scores: Optional[NDArray[np.float64]] = None


class SCCDEEstimator:
    """Self-Calibrated Constrained Dantzig-type Estimator.

    Extends the standard CDE with an adaptive scale variable tau that
    jointly penalises the l1-norm and stationarity violation. This makes
    the estimator self-calibrating: rather than selecting lambda via
    cross-validation, the SC-CDE penalises the scale of the solution
    directly with objective ``||beta||_1 + c * tau``.

    For tuning, a grid search over (lambda_scaled, c_const) pairs
    can be performed, selecting the combination that minimises
    out-of-sample portfolio variance.

    Parameters
    ----------
    lambda_grid : sequence of float, optional
        Grid of lambda values in scaled units. Default: [30, 50, 70, 90].
    c_grid : sequence of float, optional
        Grid of penalty constants. Default: [1.0, 3.0, 5.0].
    n_splits : int, default=5
        Number of folds for grid-search evaluation.
    big_M : float, default=100.0
        Big-M constant for complementarity linearisation.
    time_limit : float, optional
        Per-solve time limit in seconds.
    random_state : int, optional
        Random seed for CV fold generation.
    verbose : bool, default=False
        If True, enable debug logging.

    Examples
    --------
    >>> from cde_estimator import SCCDEEstimator
    >>> from cde_estimator.constraints import budget_constraint, liquidity_constraint, combine_constraints
    >>>
    >>> constraints = combine_constraints(
    ...     budget_constraint(p=50),
    ...     liquidity_constraint(p=50, liquid_indices=np.arange(10)),
    ... )
    >>> estimator = SCCDEEstimator()
    >>> result = estimator.fit(sigma, eta, constraints, lambda_value=90.0, c_const=3.0)
    >>> print(result.weights, result.tau)
    """

    def __init__(
        self,
        lambda_grid: Optional[Sequence[float]] = None,
        c_grid: Optional[Sequence[float]] = None,
        n_splits: int = 5,
        big_M: float = 100.0,
        time_limit: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.lambda_grid = list(lambda_grid) if lambda_grid is not None else [30.0, 50.0, 70.0, 90.0]
        self.c_grid = list(c_grid) if c_grid is not None else [1.0, 3.0, 5.0]
        self.n_splits = n_splits
        self.big_M = big_M
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    def fit(
        self,
        sigma: NDArray[np.float64],
        eta: NDArray[np.float64],
        constraints: LinearConstraints,
        lambda_value: float,
        c_const: float = 3.0,
    ) -> SCCDEResult:
        """Fit the SC-CDE for a single (lambda, c) pair.

        Parameters
        ----------
        sigma : ndarray of shape (p, p)
            Covariance matrix.
        eta : ndarray of shape (p,)
            Return signal.
        constraints : LinearConstraints
            The inequality and equality constraints.
        lambda_value : float
            Tuning parameter in scaled units.
        c_const : float, default=3.0
            Penalty constant on the scale variable tau.

        Returns
        -------
        SCCDEResult
            The fitted result.
        """
        p = constraints.p
        A, b, C, d = constraints.A, constraints.b, constraints.C, constraints.d
        k, l_eq = constraints.k, constraints.l

        lambda_max, factor = find_lambda_max(
            sigma, eta, p, A, b, k, C, d, l_eq, factor=1.0, big_M=self.big_M
        )

        w, tau = solve_self_calibrated_cde(
            sigma, eta, p, A, b, k, C, d, l_eq,
            factor=factor,
            lambda_scaled=lambda_value,
            c_const=c_const,
            big_M=self.big_M,
            time_limit=self.time_limit,
        )

        return SCCDEResult(
            weights=w,
            tau=tau,
            lambda_selected=lambda_value,
            c_selected=c_const,
            lambda_max=lambda_max,
            factor=factor,
        )

    def fit_grid(
        self,
        sigma: NDArray[np.float64],
        eta: NDArray[np.float64],
        constraints: LinearConstraints,
        returns_data: NDArray[np.float64],
        lambda_grid: Optional[Sequence[float]] = None,
        c_grid: Optional[Sequence[float]] = None,
    ) -> SCCDEResult:
        """Fit the SC-CDE with grid search over (lambda, c) pairs.

        Evaluates all combinations on out-of-sample portfolio variance
        using K-fold splitting, then refits with the best pair on
        the full sample.

        Parameters
        ----------
        sigma : ndarray of shape (p, p)
            Full-sample covariance matrix.
        eta : ndarray of shape (p,)
            Return signal.
        constraints : LinearConstraints
            Linear constraints.
        returns_data : ndarray of shape (n, p)
            Raw return data for OOS evaluation.
        lambda_grid : sequence of float, optional
            Override the instance lambda grid.
        c_grid : sequence of float, optional
            Override the instance c grid.

        Returns
        -------
        SCCDEResult
            Result fitted with the best (lambda, c) pair.
        """
        lam_grid = list(lambda_grid) if lambda_grid is not None else self.lambda_grid
        c_vals = list(c_grid) if c_grid is not None else self.c_grid

        p = constraints.p
        A, b, C, d = constraints.A, constraints.b, constraints.C, constraints.d
        k, l_eq = constraints.k, constraints.l

        # Compute lambda_max and factor on full data
        lam_max, factor = find_lambda_max(
            sigma, eta, p, A, b, k, C, d, l_eq, factor=1.0, big_M=self.big_M
        )

        # Set up CV folds
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        folds = list(kf.split(returns_data))

        # Grid search over (lambda, c)
        grid_scores = np.full((len(lam_grid), len(c_vals)), np.inf)

        for i, lam in enumerate(lam_grid):
            for j, c in enumerate(c_vals):
                fold_returns: List[NDArray] = []

                for train_idx, test_idx in folds:
                    try:
                        w, _ = solve_self_calibrated_cde(
                            sigma, eta, p, A, b, k, C, d, l_eq,
                            factor=factor,
                            lambda_scaled=lam,
                            c_const=c,
                            big_M=self.big_M,
                            time_limit=self.time_limit,
                        )
                        test_ret = returns_data[test_idx] @ w
                        fold_returns.append(test_ret)
                    except InfeasibleError:
                        logger.warning(
                            "SC-CDE (lam=%.1f, c=%.1f) infeasible on a fold",
                            lam, c,
                        )
                        fold_returns.append(np.array([np.inf]))

                all_returns = np.concatenate(fold_returns)
                grid_scores[i, j] = np.var(all_returns)

                logger.debug(
                    "SC-CDE lam=%.1f c=%.1f -> OOS var=%.6e",
                    lam, c, grid_scores[i, j],
                )

        best_idx = np.unravel_index(np.argmin(grid_scores), grid_scores.shape)
        best_lam = lam_grid[best_idx[0]]
        best_c = c_vals[best_idx[1]]
        logger.info(
            "Best (lam, c) = (%.1f, %.1f), OOS var = %.6e",
            best_lam, best_c, grid_scores[best_idx],
        )

        # Final fit on full data
        w, tau = solve_self_calibrated_cde(
            sigma, eta, p, A, b, k, C, d, l_eq,
            factor=factor,
            lambda_scaled=best_lam,
            c_const=best_c,
            big_M=self.big_M,
            time_limit=self.time_limit,
        )

        return SCCDEResult(
            weights=w,
            tau=tau,
            lambda_selected=best_lam,
            c_selected=best_c,
            lambda_max=lam_max,
            factor=factor,
            grid_scores=grid_scores,
        )
