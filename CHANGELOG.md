# Changelog

## [0.2.0] - 2026-04-16

### Added
- Self-Calibrated CDE solver (`solve_self_calibrated_cde`) with adaptive scale variable tau and factor-rescaled stationarity
- `SCCDEEstimator` high-level class with `fit()` and `fit_grid()` for grid search over (lambda, c) pairs
- `SCCDEResult` dataclass with weights, tau, lambda/c selection, and grid scores
- Tests for SC-CDE solver (constraint satisfaction, tau scaling with c, estimator integration)

## [0.1.0] - 2026-04-15

### Added
- CDE solver with KKT-based MILP formulation and automatic lambda calibration
- Equality-only CDE solver (pure LP) for precision matrix estimation
- `PrecisionMatrixEstimator` with symmetry constraints, Gaussian NLL scoring, and support recovery metrics
- Modular constraint system: budget, liquidity, volume-liquidity, gross-exposure constraints
- Cross-validation for automatic tuning parameter selection
- Full test suite (constraints, solver, precision, utilities)
