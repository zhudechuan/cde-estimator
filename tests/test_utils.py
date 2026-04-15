"""Tests for utility functions."""

import numpy as np
import pytest

from cde_estimator.utils import perturb_covariance, validate_dimensions
from cde_estimator.exceptions import InputValidationError


class TestPerturbCovariance:
    def test_positive_definite_output(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 50))
        cov = X.T @ X / 20  # rank-deficient (n < p)
        perturbed = perturb_covariance(cov)
        eigenvalues = np.linalg.eigvalsh(perturbed)
        assert np.all(eigenvalues > 0)

    def test_already_pd_unchanged_structure(self):
        cov = np.eye(5) * 2.0
        perturbed = perturb_covariance(cov)
        # For identity-like matrix, shift should be zero
        np.testing.assert_allclose(perturbed, cov, atol=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((10, 30))
        cov = X.T @ X / 10
        perturbed = perturb_covariance(cov)
        np.testing.assert_allclose(perturbed, perturbed.T, atol=1e-12)


class TestValidateDimensions:
    def test_valid(self):
        p = 10
        sigma = np.eye(p)
        eta = np.zeros(p)
        A = np.ones((2, p))
        b = np.zeros(2)
        C = np.ones((1, p))
        d = np.zeros(1)
        result = validate_dimensions(sigma, eta, A, b, C, d)
        assert result == p

    def test_eta_wrong_shape(self):
        p = 10
        sigma = np.eye(p)
        eta = np.zeros(p + 1)
        A = np.ones((1, p))
        b = np.zeros(1)
        C = np.ones((1, p))
        d = np.zeros(1)
        with pytest.raises(InputValidationError):
            validate_dimensions(sigma, eta, A, b, C, d)

    def test_Ab_mismatch(self):
        p = 10
        sigma = np.eye(p)
        eta = np.zeros(p)
        A = np.ones((2, p))
        b = np.zeros(3)  # wrong length
        C = np.ones((1, p))
        d = np.zeros(1)
        with pytest.raises(InputValidationError):
            validate_dimensions(sigma, eta, A, b, C, d)
