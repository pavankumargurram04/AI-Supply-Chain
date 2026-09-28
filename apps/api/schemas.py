"""Pydantic schemas for API requests/responses.

Kept together to avoid a sprawling schemas/ directory for a modest API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------- Auth ----------------
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    id: int
    email: str
    username: str
    role: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------- Master data ----------------
class ProductOut(BaseModel):
    id: int
    sku: str
    name: str
    description: Optional[str] = None
    category_id: Optional[int] = None

    unit_cost: float
    unit_price: float
    weight_kg: Optional[float] = None
    volume_m3: Optional[float] = None

    lifecycle_stage: str
    seasonality_class: str
    volatility_class: str

    base_demand: float
    lead_time_days: int

    class Config:
        from_attributes = True


class SupplierOut(BaseModel):
    id: int
    code: str
    name: str
    country: Optional[str]
    region: Optional[str]
    reliability_class: str

    class Config:
        from_attributes = True


class WarehouseOut(BaseModel):
    id: int
    code: str
    name: str
    region: Optional[str]
    country: Optional[str]
    capacity_units: float

    class Config:
        from_attributes = True


# ---------------- Inventory / decisions ----------------
class DecisionRequestIn(BaseModel):
    product_sku: str
    warehouse_code: str
    service_level: float = 0.95
    horizon_days: int = 30
    demand_multiplier: float = 1.0
    lead_time_override_days: Optional[float] = None
    lead_time_std_override: Optional[float] = None
    inventory_override: Optional[float] = None
    budget: Optional[float] = None
    max_suppliers: Optional[int] = None
    preferred_supplier_codes: Optional[List[str]] = None
    seed: int = 42
    scenario_id: Optional[str] = None


class DecisionTraceOut(BaseModel):
    trace_id: str
    run_at: str
    product_sku: str
    warehouse_code: str
    final_quantity: float
    total_cost: float
    expected_service_level: float
    stockout_probability: float
    safety_stock: float
    reorder_point: float
    confidence: float
    guardrail_violations: List[str]
    explanation: str
    full_trace: Dict[str, Any]


# ---------------- Forecasts ----------------
class ForecastOut(BaseModel):
    product_sku: str
    warehouse_code: str
    horizon: int
    origin_date: str
    forecast_dates: List[str]
    point: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    quantiles: Optional[Dict[str, List[float]]] = None
    model_version: str
    feature_version: str
    seed: int


# ---------------- Supplier risk ----------------
class SupplierRiskOut(BaseModel):
    supplier_code: str
    risk_probability: float
    risk_label: str
    lead_time_days: float
    unit_price: float
    moq: int
    is_preferred: bool
    model_version: Optional[str] = None


# ---------------- Approvals ----------------
class OverrideRequest(BaseModel):
    new_quantity: float = Field(..., ge=0.0)
    reason: str = Field(..., min_length=3, max_length=500)


class ApproveRequest(BaseModel):
    notes: Optional[str] = None


class RecommendationListItem(BaseModel):
    id: int
    trace_id: str
    product_id: int
    warehouse_id: int
    status: str
    final_quantity: Optional[float]
    expected_cost: Optional[float]
    expected_service_level: Optional[float]
    stockout_probability: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------- Agents ----------------
class AgentRunIn(BaseModel):
    product_sku: str
    warehouse_code: str
    service_level: float = 0.95
    horizon_days: int = 30
    demand_multiplier: float = 1.0
    lead_time_override_days: Optional[float] = None
    inventory_override: Optional[float] = None
    seed: int = 42


class AgentMessageOut(BaseModel):
    message_id: str
    trace_id: str
    agent: str
    timestamp: str
    status: str
    confidence: Optional[float]
    payload: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    model_versions: Dict[str, str] = Field(default_factory=dict)