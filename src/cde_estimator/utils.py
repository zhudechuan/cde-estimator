"""Utility functions for the CDE estimator library."""

import numpy as np
from numpy.typing import NDArray


def perturb_covariance(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply spectral perturbation to ensure positive definiteness.

    Shifts the covariance matrix by adding a scaled identity matrix,
    using the Ledoit-Wolf-style correction based on the eigenvalue spread.

    Parameters
    ----------
    cov : ndarray of shape (p, p)
        Sample covariance matrix (symmetric, positive semi-definite).

    Returns
    -------
    ndarray of shape (p, p)
        Perturbed covariance matrix guaranteed to be positive definite.
    """
    p = cov.shape[0]
    eigenvalues = np.linalg.eigvalsh(cov)
    shift = max(eigenvalues[-1] - p * eigenvalues[0], 0.0) / (p - 1)
    return cov + np.eye(p) * shift


def validate_dimensions(
    sigma: NDArray[np.float64],
    eta: NDArray[np.float64],
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    C: NDArray[np.float64],
    d: NDArray[np.float64],
) -> int:
    """Validate and return the problem dimension p.

    Parameters
    ----------
    sigma : ndarray of shape (p, p)
        Covariance matrix.
    eta : ndarray of shape (p,)
        Expected return vector.
    A : ndarray of shape (k, p)
        Inequality constraint matrix.
    b : ndarray of shape (k,)
        Inequality constraint RHS.
    C : ndarray of shape (l, p)
        Equality constraint matrix.
    d : ndarray of shape (l,)
        Equality constraint RHS.

    Returns
    -------
    int
        Problem dimension p.

    Raises
    ------
    InputValidationError
        If dimensions are inconsistent.
    """
    from .exceptions import InputValidationError

    p = sigma.shape[0]
    if sigma.shape != (p, p):
        raise InputValidationError(f"sigma must be square, got shape {sigma.shape}")
    if eta.shape != (p,):
        raise InputValidationError(
            f"eta must have shape ({p},), got {eta.shape}"
        )
    if A.ndim != 2 or A.shape[1] != p:
        raise InputValidationError(
            f"A must have shape (k, {p}), got {A.shape}"
        )
    if b.shape[0] != A.shape[0]:
        raise InputValidationError(
            f"b length {b.shape[0]} != A rows {A.shape[0]}"
        )
    if C.ndim != 2 or C.shape[1] != p:
        raise InputValidationError(
            f"C must have shape (l, {p}), got {C.shape}"
        )
    if d.shape[0] != C.shape[0]:
        raise InputValidationError(
            f"d length {d.shape[0]} != C rows {C.shape[0]}"
        )
    return p
