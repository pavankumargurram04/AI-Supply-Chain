"""Typed exceptions used across the platform.

Every domain error inherits from `AppError` so the API layer can translate
it uniformly. Do not use bare `Exception` for expected failure modes.
"""

from __future__ import annotations

from typing import Any, Optional


class AppError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    error_code: str = "app_error"

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(AppError):
    status_code = 500
    error_code = "configuration_error"


class ValidationError(AppError):
    status_code = 422
    error_code = "validation_error"


class NotFoundError(AppError):
    status_code = 404
    error_code = "not_found"


class ConflictError(AppError):
    status_code = 409
    error_code = "conflict"


class AuthenticationError(AppError):
    status_code = 401
    error_code = "authentication_error"


class AuthorizationError(AppError):
    status_code = 403
    error_code = "authorization_error"


class DatabaseError(AppError):
    status_code = 500
    error_code = "database_error"


class ModelError(AppError):
    status_code = 500
    error_code = "model_error"


class DataQualityError(AppError):
    status_code = 422
    error_code = "data_quality_error"


class OptimizationError(AppError):
    status_code = 500
    error_code = "optimization_error"


class ExternalServiceError(AppError):
    status_code = 502
    error_code = "external_service_error"


class RateLimitError(AppError):
    status_code = 429
    error_code = "rate_limit_error"