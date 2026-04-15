"""Custom exceptions for the CDE estimator library."""


class CDEError(Exception):
    """Base exception for CDE estimator errors."""
    pass


class InfeasibleError(CDEError):
    """Raised when the optimization problem is infeasible."""

    def __init__(self, stage: str, details: str = ""):
        self.stage = stage
        self.details = details
        msg = f"Infeasible problem at stage: {stage}"
        if details:
            msg += f" ({details})"
        super().__init__(msg)


class SolverError(CDEError):
    """Raised when the solver encounters an unexpected error."""
    pass


class InputValidationError(CDEError):
    """Raised when input parameters fail validation."""
    pass
