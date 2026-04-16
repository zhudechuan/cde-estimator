"""Precision matrix estimation via the Constrained Dantzig-type Estimator.

Estimates the precision matrix (inverse covariance) Omega = Sigma^{-1}
for a p-variate distribution by solving the Dantzig-type system

    min ||vec(Omega)||_1
    s.t.  ||(I_p \\otimes Sigma_hat) vec(Omega) - vec(I_p)||_inf <= lambda
          Omega = Omega^T

The symmetry constraint is encoded as an equality constraint on the
vectorised unknowns. Since there are no inequality constraints, the
problem is a pure LP — much faster than the MILP formulation used for
portfolio selection with inequality constraints.

Reference
---------
Pun, C. S. and Zhu, D. (2024). Constrained Dantzig-type Estimator with
inequality constraints for high-dimensional sparse learning.

Example
-------
>>> import numpy as np
>>> from cde_estimator.precision import PrecisionMatrixEstimator
>>>
>>> rng = np.random.default_rng(42)
>>> p = 10
>>> Sigma_true = ...  # true covariance
>>> X = rng.multivariate_normal(np.zeros(p), Sigma_true, size=100)
>>> data = pd.DataFrame(X)
>>>
>>> estimator = PrecisionMatrixEstimator()
>>> result = estimator.fit_cv(data)
>>> print(result.omega)  # estimated precision matrix
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from .exceptions import InfeasibleError
from .solver import solve_cde_equality

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linear algebra helpers for the vectorised precision matrix problem
# ---------------------------------------------------------------------------


def symmetry_constraint_matrix(p: int) -> NDArray[np.float64]:
    """Build the equality constraint matrix enforcing Omega = Omega^T.

    For a p x p matrix vectorised in column-major (Fortran) order, this
    produces a matrix A of shape (p*(p-1)/2, p^2) such that
    A @ vec(Omega) = 0 if and only if Omega is symmetric.

    Parameters
    ----------
    p : int
        Matrix dimension.

    Returns
    -------
    ndarray of shape (p*(p-1)//2, p*p)
        Symmetry constraint matrix.
    """
    m = p * (p - 1) // 2
    A = np.zeros((m, p * p), dtype=np.float64)
    r = 0
    for j in range(p):
        for i in range(j):
            A[r, i + j * p] = 1.0   # Omega[i, j]
            A[r, j + i * p] = -1.0  # Omega[j, i]
            r += 1
    return A


def sigma_tilde(Sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the generalised covariance Sigma_tilde = I_p (kron) Sigma.

    This is the block-diagonal matrix appearing in the vectorised
    stationarity condition for precision matrix estimation.

    Parameters
    ----------
    Sigma : ndarray of shape (p, p)
        Sample covariance matrix.

    Returns
    -------
    ndarray of shape (p^2, p^2)
        Kronecker product I_p (kron) Sigma.
    """
    p = Sigma.shape[0]
    return np.kron(np.eye(p, dtype=Sigma.dtype), Sigma)


def vec_identity(p: int) -> NDArray[np.float64]:
    """Return vec(I_p) — the identity matrix stacked column by column.

    Parameters
    ----------
    p : int
        Matrix dimension.

    Returns
    -------
    ndarray of shape (p*p,)
        Vectorised identity matrix in Fortran (column-major) order.
    """
    return np.eye(p, dtype=np.float64).reshape(-1, order="F")


def unvec(w: NDArray[np.float64], p: int) -> NDArray[np.float64]:
    """Reshape a vectorised matrix back to (p, p) in column-major order.

    Parameters
    ----------
    w : ndarray of shape (p*p,)
        Vectorised matrix.
    p : int
        Matrix dimension.

    Returns
    -------
    ndarray of shape (p, p)
        Reshaped matrix.
    """
    return w.reshape((p, p), order="F")


def gaussian_nll(
    S: NDArray[np.float64],
    Omega: NDArray[np.float64],
    eps: float = 1e-6,
) -> float:
    """Gaussian negative log-likelihood score.

    Computes -log det(Omega) + trace(S @ Omega) after projecting Omega
    to the nearest positive-definite matrix (eigenvalue flooring).

    Parameters
    ----------
    S : ndarray of shape (p, p)
        Validation (or true) covariance matrix.
    Omega : ndarray of shape (p, p)
        Estimated precision matrix.
    eps : float, default=1e-6
        Minimum eigenvalue floor to ensure positive definiteness.

    Returns
    -------
    float
        NLL score (lower is better).
    """
    Om = 0.5 * (Omega + Omega.T)
    eigenvalues, eigenvectors = np.linalg.eigh(Om)
    eigenvalues = np.clip(eigenvalues, eps, None)
    Om_pd = (eigenvectors * eigenvalues) @ eigenvectors.T
    logdet = np.sum(np.log(eigenvalues))
    return float(-logdet + np.trace(S @ Om_pd))


def frobenius_error(
    Omega_true: NDArray[np.float64],
    Omega_hat: NDArray[np.float64],
) -> float:
    """Frobenius norm of the estimation error.

    Parameters
    ----------
    Omega_true : ndarray of shape (p, p)
        True precision matrix.
    Omega_hat : ndarray of shape (p, p)
        Estimated precision matrix.

    Returns
    -------
    float
        ||Omega_true - Omega_hat||_F
    """
    return float(np.linalg.norm(Omega_true - Omega_hat, "fro"))


def support_recovery_metrics(
    Omega_true: NDArray[np.float64],
    Omega_hat: NDArray[np.float64],
    threshold: float = 1e-4,
) -> dict:
    """Compute support recovery metrics (TPR, FPR, F1) for sparsity pattern.

    Parameters
    ----------
    Omega_true : ndarray of shape (p, p)
        True precision matrix.
    Omega_hat : ndarray of shape (p, p)
        Estimated precision matrix.
    threshold : float, default=1e-4
        Entries with absolute value below this are treated as zero.

    Returns
    -------
    dict
        Dictionary with keys 'tpr', 'fpr', 'f1', 'precision', 'recall'.
    """
    true_support = np.abs(Omega_true) > threshold
    est_support = np.abs(Omega_hat) > threshold

    # Exclude diagonal
    mask = ~np.eye(Omega_true.shape[0], dtype=bool)
    true_nz = true_support[mask]
    est_nz = est_support[mask]

    tp = np.sum(true_nz & est_nz)
    fp = np.sum(~true_nz & est_nz)
    fn = np.sum(true_nz & ~est_nz)
    tn = np.sum(~true_nz & ~est_nz)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * tpr / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )

    return {
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": tpr,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Data generation for simulation studies
# ---------------------------------------------------------------------------


def generate_sparse_covariance(
    p: int,
    c0: int = 5,
    probability: float = 0.2,
    model: str = "1",
    scale: float = 0.04,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """Generate a sparse covariance matrix for simulation.

    Parameters
    ----------
    p : int
        Matrix dimension.
    c0 : int, default=5
        Number of "dense" rows/columns in model 1.
    probability : float, default=0.2
        Bernoulli probability for non-zero off-diagonal entries (model 1).
    model : str, default="1"
        Covariance structure:
        - "1": Sparse with dense block + Bernoulli entries, standardised.
        - "2": AR(1) with rho=0.8 (banded, naturally sparse inverse).
    scale : float, default=0.04
        Variance scaling factor (0.2^2 = 0.04 by default).
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray of shape (p, p)
        Covariance matrix.
    """
    rng = np.random.default_rng(seed)

    if model == "1":
        if p <= c0:
            raise ValueError(f"p must be > c0={c0}, got p={p}")
        B = (rng.random((p, p)) < probability).astype(np.float64) * 0.5
        B[c0:, c0:] = 0.5
        np.fill_diagonal(B, 1.0)
        B = np.triu(B, 1).T + np.triu(B)

        eigenvalues = np.linalg.eigvalsh(B)
        delta = max(-eigenvalues.min(), 0) + 0.05
        gamma_inv = (B + delta * np.eye(p)) / (1 + delta)
        gamma = np.linalg.inv(gamma_inv)

        # Standardise to unit diagonal
        D_inv = np.diag(1.0 / np.sqrt(np.diag(gamma)))
        gamma = D_inv @ gamma @ D_inv

    elif model == "2":
        gamma = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                gamma[i, j] = 0.8 ** abs(i - j)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    return scale * gamma


# ---------------------------------------------------------------------------
# High-level estimator
# ---------------------------------------------------------------------------


@dataclass
class PrecisionMatrixResult:
    """Result of precision matrix estimation.

    Attributes
    ----------
    omega : ndarray of shape (p, p)
        Estimated precision matrix.
    lambda_selected : float
        Tuning parameter used.
    cv_scores : ndarray or None
        Cross-validation scores if CV was used.
    """

    omega: NDArray[np.float64]
    lambda_selected: float
    cv_scores: Optional[NDArray[np.float64]] = None


class PrecisionMatrixEstimator:
    """Estimate a sparse precision matrix via the CDE.

    Uses the equality-only CDE formulation where the symmetry of Omega
    is enforced through linear constraints on vec(Omega).

    Parameters
    ----------
    lambda_grid : sequence of float, optional
        Grid of lambda candidates. Default covers [0, 1] in steps of 0.05.
    n_splits : int, default=5
        Number of CV folds.
    scoring : str, default="gaussian_nll"
        CV scoring function. Currently supports "gaussian_nll".
    time_limit : float, optional
        Per-solve time limit in seconds.
    random_state : int, optional
        Random seed for CV folds.
    verbose : bool, default=False
        Enable debug logging.

    Examples
    --------
    >>> import pandas as pd
    >>> from cde_estimator.precision import PrecisionMatrixEstimator
    >>>
    >>> data = pd.DataFrame(np.random.randn(100, 10))
    >>> est = PrecisionMatrixEstimator()
    >>> result = est.fit_cv(data)
    >>> print(result.omega.shape)  # (10, 10)
    >>> print(result.lambda_selected)
    """

    def __init__(
        self,
        lambda_grid: Optional[Sequence[float]] = None,
        n_splits: int = 5,
        scoring: str = "gaussian_nll",
        time_limit: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        if lambda_grid is None:
            self.lambda_grid = [i / 20 for i in range(20, -1, -1)]
        else:
            self.lambda_grid = list(lambda_grid)

        self.n_splits = n_splits
        self.scoring = scoring
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    @staticmethod
    def _build_problem(
        data: pd.DataFrame,
    ) -> tuple:
        """Build the vectorised CDE problem from a data frame.

        Returns (sigma_tilde_mat, eta, p_vec, A, b, k, p_orig).
        """
        S = data.cov().values
        p = S.shape[0]
        eta = vec_identity(p)
        A = symmetry_constraint_matrix(p)
        sig_tilde = sigma_tilde(S)
        p_vec = p ** 2
        b = np.zeros(A.shape[0])
        k = A.shape[0]
        return sig_tilde, eta, p_vec, A, b, k, p

    def fit(
        self,
        data: pd.DataFrame,
        lambda_value: float,
    ) -> PrecisionMatrixResult:
        """Fit the precision matrix for a specific lambda.

        Parameters
        ----------
        data : DataFrame of shape (n, p)
            Sample data (observations x variables).
        lambda_value : float
            Tuning parameter.

        Returns
        -------
        PrecisionMatrixResult
        """
        sig_tilde, eta, p_vec, A, b, k, p = self._build_problem(data)
        factor = 1.0  # no rescaling needed for precision matrix

        w = solve_cde_equality(
            sig_tilde, eta, p_vec, A, b, k,
            factor=factor,
            lambda_scaled=lambda_value,
            time_limit=self.time_limit,
        )
        omega = unvec(w, p)

        return PrecisionMatrixResult(
            omega=omega,
            lambda_selected=lambda_value,
        )

    def fit_cv(
        self,
        data: pd.DataFrame,
        lambda_grid: Optional[Sequence[float]] = None,
    ) -> PrecisionMatrixResult:
        """Fit with K-fold cross-validation for lambda selection.

        For each fold, the CDE is solved on the training covariance, and
        the estimated precision matrix is scored against the validation
        covariance using the Gaussian negative log-likelihood.

        Parameters
        ----------
        data : DataFrame of shape (n, p)
            Sample data.
        lambda_grid : sequence of float, optional
            Override lambda grid for this call.

        Returns
        -------
        PrecisionMatrixResult
        """
        grid = list(lambda_grid) if lambda_grid is not None else self.lambda_grid
        p = data.shape[1]

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        folds = list(kf.split(data))

        cv_scores = np.full(len(grid), np.inf)

        for idx, lam in enumerate(grid):
            fold_scores: List[float] = []

            for train_idx, test_idx in folds:
                data_train = data.iloc[train_idx]
                data_test = data.iloc[test_idx]

                sig_tilde_mat, eta, p_vec, A, b_vec, k, _ = self._build_problem(
                    data_train
                )

                try:
                    w = solve_cde_equality(
                        sig_tilde_mat, eta, p_vec, A, b_vec, k,
                        factor=1.0,
                        lambda_scaled=lam,
                        time_limit=self.time_limit,
                    )
                    omega = unvec(w, p)

                    if self.scoring == "gaussian_nll":
                        S_val = data_test.cov().values
                        fold_scores.append(gaussian_nll(S_val, omega))
                    else:
                        raise ValueError(f"Unknown scoring: {self.scoring}")

                except InfeasibleError:
                    logger.warning("Lambda %.4f infeasible on fold, skipping", lam)
                    continue

            if fold_scores:
                cv_scores[idx] = np.mean(fold_scores)

            logger.debug("Lambda %.4f -> CV score %.6e", lam, cv_scores[idx])

        best_idx = int(np.nanargmin(cv_scores))
        best_lambda = grid[best_idx]
        logger.info(
            "Best lambda: %.4f (CV score: %.6e)", best_lambda, cv_scores[best_idx]
        )

        # Final fit on full data
        sig_tilde_mat, eta, p_vec, A, b_vec, k, _ = self._build_problem(data)
        w = solve_cde_equality(
            sig_tilde_mat, eta, p_vec, A, b_vec, k,
            factor=1.0,
            lambda_scaled=best_lambda,
            time_limit=self.time_limit,
        )
        omega = unvec(w, p)

        return PrecisionMatrixResult(
            omega=omega,
            lambda_selected=best_lambda,
            cv_scores=cv_scores,
        )
