# Changelog

## [0.1.0] - 2026-04-15

### Added
- CDE solver with KKT-based MILP formulation and automatic lambda calibration
- Equality-only CDE solver (pure LP) for precision matrix estimation
- `PrecisionMatrixEstimator` with symmetry constraints, Gaussian NLL scoring, and support recovery metrics
- Modular constraint system: budget, liquidity, volume-liquidity, gross-exposure constraints
- Cross-validation for automatic tuning parameter selection
- Full test suite (constraints, solver, precision, utilities)
