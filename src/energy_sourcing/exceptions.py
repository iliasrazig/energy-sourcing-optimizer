"""Exception hierarchy: callers can catch the base class or a precise failure."""


class EnergySourcingError(Exception):
    """Base class for all errors raised by this package."""


class DataValidationError(EnergySourcingError):
    """The dataset violates a schema or quality expectation."""


class NotFittedError(EnergySourcingError):
    """A model was used before being trained."""


class QualityGateError(EnergySourcingError):
    """A trained model did not meet the minimum quality required for release."""


class ArtifactNotFoundError(EnergySourcingError):
    """A required model artifact is missing from the model store."""


class DayNotAvailableError(EnergySourcingError):
    """No out-of-sample forecast can be produced for the requested delivery day."""


class OptimizationError(EnergySourcingError):
    """The sourcing optimisation problem could not be solved."""
