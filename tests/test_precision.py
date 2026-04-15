"""Tests for precision matrix estimation helpers (no CPLEX needed)."""

import numpy as np
import pytest

from cde_estimator.precision import (
    frobenius_error,
    gaussian_nll,
    generate_sparse_covariance,
    support_recovery_metrics,
    symmetry_constraint_matrix,
    sigma_tilde,
    unvec,
    vec_identity,
)


class TestVecIdentity:
    def test_shape(self):
        v = vec_identity(5)
        assert v.shape == (25,)

    def test_values(self):
        v = vec_identity(3)
        # Column-major vec of I_3: [1,0,0, 0,1,0, 0,0,1]
        expected = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
        np.testing.assert_array_equal(v, expected)


class TestUnvec:
    def test_roundtrip(self):
        p = 4
        M = np.arange(16, dtype=float).reshape(4, 4, order="F")
        v = M.reshape(-1, order="F")
        np.testing.assert_array_equal(unvec(v, p), M)


class TestSymmetryConstraint:
    def test_shape(self):
        p = 5
        A = symmetry_constraint_matrix(p)
        assert A.shape == (p * (p - 1) // 2, p * p)

    def test_symmetric_matrix_satisfies(self):
        p = 4
        A = symmetry_constraint_matrix(p)
        # Symmetric matrix
        M = np.array([[1, 2, 3, 4],
                       [2, 5, 6, 7],
                       [3, 6, 8, 9],
                       [4, 7, 9, 10]], dtype=float)
        v = M.reshape(-1, order="F")
        np.testing.assert_allclose(A @ v, 0, atol=1e-12)

    def test_asymmetric_fails(self):
        p = 3
        A = symmetry_constraint_matrix(p)
        M = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=float)
        v = M.reshape(-1, order="F")
        assert np.linalg.norm(A @ v) > 0


class TestSigmaTilde:
    def test_shape(self):
        S = np.eye(4) * 2.0
        result = sigma_tilde(S)
        assert result.shape == (16, 16)

    def test_kronecker_identity(self):
        p = 3
        S = np.eye(p) * 5.0
        result = sigma_tilde(S)
        expected = np.kron(np.eye(p), S)
        np.testing.assert_allclose(result, expected)


class TestGaussianNLL:
    def test_identity_case(self):
        p = 5
        S = np.eye(p)
        Omega = np.eye(p)
        # -log det(I) + trace(I @ I) = 0 + p = p
        nll = gaussian_nll(S, Omega)
        np.testing.assert_allclose(nll, p, atol=1e-4)

    def test_scaled_case(self):
        p = 3
        S = np.eye(p) * 2.0
        Omega = np.eye(p) * 0.5  # inverse of S
        # -log det(0.5*I) + trace(2I @ 0.5I) = -p*log(0.5) + p = p*log(2) + p
        expected = p * np.log(2) + p
        nll = gaussian_nll(S, Omega)
        np.testing.assert_allclose(nll, expected, atol=1e-4)


class TestFrobeniusError:
    def test_same_matrix(self):
        M = np.random.randn(5, 5)
        assert frobenius_error(M, M) == pytest.approx(0.0)

    def test_known_value(self):
        A = np.eye(3)
        B = np.zeros((3, 3))
        # ||I - 0||_F = sqrt(3)
        assert frobenius_error(A, B) == pytest.approx(np.sqrt(3))


class TestSupportRecovery:
    def test_perfect_recovery(self):
        p = 5
        Omega = np.eye(p)
        Omega[0, 1] = Omega[1, 0] = 0.5
        metrics = support_recovery_metrics(Omega, Omega)
        assert metrics["tpr"] == pytest.approx(1.0)
        assert metrics["fpr"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_zero_estimate(self):
        p = 5
        Omega_true = np.eye(p)
        Omega_true[0, 1] = Omega_true[1, 0] = 0.5
        Omega_hat = np.zeros((p, p))
        metrics = support_recovery_metrics(Omega_true, Omega_hat)
        assert metrics["tpr"] == pytest.approx(0.0)
        assert metrics["fpr"] == pytest.approx(0.0)


class TestGenerateSparseCovariance:
    def test_model1_shape(self):
        cov = generate_sparse_covariance(20, seed=42)
        assert cov.shape == (20, 20)

    def test_model1_symmetric(self):
        cov = generate_sparse_covariance(15, seed=0)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_model1_pd(self):
        cov = generate_sparse_covariance(20, seed=99)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_model2_shape(self):
        cov = generate_sparse_covariance(10, model="2")
        assert cov.shape == (10, 10)

    def test_model2_ar1_structure(self):
        cov = generate_sparse_covariance(5, model="2", scale=1.0)
        # (0,1) entry should be 0.8^1
        np.testing.assert_allclose(cov[0, 1], 0.8, atol=1e-12)

    def test_small_p_raises(self):
        with pytest.raises(ValueError, match="p must be"):
            generate_sparse_covariance(3, c0=5)
