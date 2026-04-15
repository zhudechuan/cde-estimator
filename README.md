# CDE Estimator

**Constrained Dantzig-type Estimator for high-dimensional sparse learning.**

A Python library implementing the CDE framework from Pun & Zhu (2024), which solves sparse estimation problems under general linear inequality and equality constraints via mixed-integer linear programming (MILP) or linear programming (LP).

## Features

- **CDE solver** with KKT-based MILP formulation and automatic lambda calibration
- **Equality-only CDE solver** — pure LP for problems like precision matrix estimation (no binary variables, much faster)
- **Precision matrix estimation** — dedicated `PrecisionMatrixEstimator` with symmetry constraints, Gaussian NLL scoring, and support recovery metrics
- **Modular constraint system** — compose budget, liquidity, and exposure constraints or bring your own
- **Cross-validation** for automatic tuning parameter selection
- **Clean API** with type hints, numpy-style docstrings, and proper error handling

## Requirements

- Python >= 3.9
- IBM CPLEX (via `docplex`) — [free academic licence available](https://www.ibm.com/academic)
- NumPy, pandas, scikit-learn

## Installation

```bash
pip install cde-estimator
```

Or from source:

```bash
git clone https://github.com/dechuan-zhu/cde-estimator.git
cd cde-estimator
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from cde_estimator import CDEEstimator
from cde_estimator.constraints import (
    budget_constraint,
    liquidity_constraint,
    combine_constraints,
)

# Prepare data
returns = np.random.randn(252, 50) * 0.01  # (n_obs, n_assets)
sigma = np.cov(returns, rowvar=False, ddof=1)
eta = returns.mean(axis=0)
p = sigma.shape[0]

# Build constraints: fully invested + liquidity tiers
constraints = combine_constraints(
    budget_constraint(p, total=1.0),
    liquidity_constraint(p, liquid_indices=np.arange(10)),
)

# Fit with a specific lambda
estimator = CDEEstimator()
result = estimator.fit(sigma, eta, constraints, lambda_value=90.0)
print("Weights:", result.weights)
print("L1 norm:", np.abs(result.weights).sum())

# Fit with cross-validation
result_cv = estimator.fit_cv(sigma, eta, constraints, returns)
print("Best lambda:", result_cv.lambda_selected)
print("CV scores:", result_cv.cv_scores)
```

## Constraint Builders

The library provides composable constraint factories:

| Function | Description |
|---|---|
| `budget_constraint(p, total)` | Sum-to-one (or zero for transactions) |
| `liquidity_constraint(p, liquid_indices, ...)` | Tiered illiquidity penalty |
| `volume_liquidity_constraint(p, volume, ...)` | Volume-based liquidity tiers |
| `gross_exposure_constraint(p, sigma, w0, ...)` | Per-asset exposure bounds |
| `combine_constraints(*constraints)` | Stack multiple constraints |

You can also pass custom `LinearConstraints(A, b, C, d)` directly.

## Precision Matrix Estimation

The library includes a dedicated module for estimating sparse precision matrices
(inverse covariance), which uses the equality-only CDE formulation:

```python
import pandas as pd
from cde_estimator.precision import PrecisionMatrixEstimator, generate_sparse_covariance

# Generate synthetic data
Sigma = generate_sparse_covariance(p=20, model="1", seed=42)
rng = np.random.default_rng(42)
X = rng.multivariate_normal(np.zeros(20), Sigma, size=100)
data = pd.DataFrame(X)

# Estimate precision matrix with cross-validation
estimator = PrecisionMatrixEstimator(random_state=42)
result = estimator.fit_cv(data)

print("Selected lambda:", result.lambda_selected)
print("Omega shape:", result.omega.shape)
```

The precision module also provides evaluation utilities:

```python
from cde_estimator.precision import gaussian_nll, frobenius_error, support_recovery_metrics

Omega_true = np.linalg.inv(Sigma)
print("NLL:", gaussian_nll(Sigma, result.omega))
print("Frobenius error:", frobenius_error(Omega_true, result.omega))
print("Support recovery:", support_recovery_metrics(Omega_true, result.omega))
```

## Low-Level API

For direct access to the solvers:

```python
# Inequality-constrained CDE (MILP)
from cde_estimator import find_lambda_max, solve_cde

lambda_max, factor = find_lambda_max(sigma, eta, p, A, b, k, C, d, l)
w = solve_cde(sigma, eta, p, A, b, k, C, d, l, factor, lambda_scaled=90.0)

# Equality-only CDE (LP) — e.g., for precision matrix estimation
from cde_estimator import find_lambda_max_equality, solve_cde_equality

lambda_max, factor = find_lambda_max_equality(sigma_tilde, eta, p_sq, A_sym, b_sym, k_sym)
w = solve_cde_equality(sigma_tilde, eta, p_sq, A_sym, b_sym, k_sym, factor, lambda_scaled=0.5)
```

## Citation

If you use this library in your research, please cite:

```bibtex
@article{pun2024cde,
  title={Constrained Dantzig-type Estimator with Inequality Constraints
         for High-Dimensional Sparse Learning},
  author={Pun, Chi Seng and Zhu, Dechuan},
  year={2024},
}
```

## License

MIT
