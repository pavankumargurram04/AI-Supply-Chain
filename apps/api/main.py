"""FastAPI application factory.

The app is intentionally thin — all business logic lives in
`decision_engine`, `agents`, `ml/*`, and `db/repositories`. Routers only:
    1) validate input,
    2) call the right service,
    3) shape the response.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.exception_handlers import install_exception_handlers
from apps.api.middleware import RequestContextMiddleware


# --- Routers ---
from apps.api.routers import (
    agents as agents_router,
    audit as audit_router,
    auth as auth_router,
    forecasting as forecasting_router,
    health as health_router,
    inventory as inventory_router,
    models as models_router,
    monitoring as monitoring_router,
    orders as orders_router,
    procurement as procurement_router,
    products as products_router,
    reports as reports_router,
    scenarios as scenarios_router,
    shipments as shipments_router,
    supplier_risk as supplier_risk_router,
    suppliers as suppliers_router,
    training as training_router,
    users as users_router,
    warehouses as warehouses_router,
    
)
from core.config import get_settings
from core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    log.info("api.startup", extra={
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
    })
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI Supply Chain Enterprise — decision-intelligence platform.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS for the local React dev server.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    install_exception_handlers(app)

    prefix = settings.API_V1_PREFIX

    # --- Health lives at root, no prefix ---
    app.include_router(health_router.router)

    # --- Everything else under /api/v1 ---
    app.include_router(auth_router.router, prefix=prefix)
    app.include_router(users_router.router, prefix=prefix)
    app.include_router(products_router.router, prefix=prefix)
    app.include_router(suppliers_router.router, prefix=prefix)
    app.include_router(warehouses_router.router, prefix=prefix)
    app.include_router(inventory_router.router, prefix=prefix)
    app.include_router(forecasting_router.router, prefix=prefix)
    app.include_router(supplier_risk_router.router, prefix=prefix)
    app.include_router(procurement_router.router, prefix=prefix)
    app.include_router(scenarios_router.router, prefix=prefix)
    app.include_router(agents_router.router, prefix=prefix)
    app.include_router(models_router.router, prefix=prefix)
    app.include_router(training_router.router, prefix=prefix)
    app.include_router(monitoring_router.router, prefix=prefix)
    app.include_router(reports_router.router, prefix=prefix)
    app.include_router(shipments_router.router, prefix=prefix)
    app.include_router(orders_router.router, prefix=prefix)
    app.include_router(audit_router.router, prefix=prefix)

    @app.get("/")
    def root() -> dict:
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "api_prefix": prefix,
        }

    return app


app = create_app()