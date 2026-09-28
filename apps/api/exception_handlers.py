"""Map AppError subclasses and validation errors to structured JSON."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.errors import AppError
from core.logging import get_logger

log = get_logger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        log.warning("http.app_error", extra={
            "request_id": _request_id(request),
            "error_code": exc.error_code,
            "error_message": exc.message,
        })
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "validation_error",
                "message": "Request payload failed validation",
                "details": {"errors": exc.errors()},
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": "http_error",
                "message": exc.detail if isinstance(exc.detail, str) else "HTTP error",
                "details": {} if isinstance(exc.detail, str) else {"detail": exc.detail},
                "request_id": _request_id(request),
            },
        )


