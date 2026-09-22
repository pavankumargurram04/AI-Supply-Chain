"""Enumerations shared across the platform.

All enums inherit from `str` so they serialise to JSON cleanly and can be
persisted directly as VARCHAR columns without a custom SQLAlchemy type.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum base with a friendly __str__."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --- Users / security -----------------------------------------------------
class UserRole(StrEnum):
    ADMIN = "admin"
    ANALYST = "analyst"
    PROCUREMENT_MANAGER = "procurement_manager"
    SUPPLY_CHAIN_MANAGER = "supply_chain_manager"
    REVIEWER = "reviewer"


# --- Master data ----------------------------------------------------------
class LifecycleStage(StrEnum):
    INTRODUCTION = "introduction"
    GROWTH = "growth"
    MATURITY = "maturity"
    DECLINE = "decline"
    DISCONTINUED = "discontinued"


class SeasonalityClass(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class VolatilityClass(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ReliabilityClass(StrEnum):
    PREMIUM = "premium"
    RELIABLE = "reliable"
    STANDARD = "standard"
    POOR = "poor"


# --- Inventory / shipments -----------------------------------------------
class MovementType(StrEnum):
    RECEIPT = "receipt"
    ISSUE = "issue"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    ADJUSTMENT = "adjustment"
    RETURN = "return"
    DAMAGE = "damage"
    RESERVATION = "reservation"
    RESERVATION_RELEASE = "reservation_release"
    STOCKOUT = "stockout"


class ShipmentStatus(StrEnum):
    PLANNED = "planned"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    ARRIVED = "arrived"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class ShipmentEventType(StrEnum):
    CREATED = "created"
    DISPATCHED = "dispatched"
    IN_TRANSIT_UPDATE = "in_transit_update"
    CUSTOMS = "customs"
    DELAY_REPORTED = "delay_reported"
    DELIVERED = "delivered"
    DAMAGED = "damaged"
    CANCELLED = "cancelled"


class OrderStatus(StrEnum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ORDERED = "ordered"
    PARTIALLY_RECEIVED = "partially_received"
    RECEIVED = "received"
    CANCELLED = "cancelled"


# --- Supplier events ------------------------------------------------------
class SupplierEventType(StrEnum):
    DELAY = "delay"
    EARLY = "early"
    QUALITY_INCIDENT = "quality_incident"
    CAPACITY_OUTAGE = "capacity_outage"
    PRICE_CHANGE = "price_change"
    FILL_RATE_DROP = "fill_rate_drop"
    ON_TIME_MISS = "on_time_miss"
    RECOVERY = "recovery"


class QualityEventType(StrEnum):
    INSPECTION_PASS = "inspection_pass"
    INSPECTION_FAIL = "inspection_fail"
    DEFECT_FOUND = "defect_found"
    RETURN = "return"
    REWORK = "rework"
    SCRAP = "scrap"


# --- ML lifecycle ---------------------------------------------------------
class ModelFamily(StrEnum):
    DEMAND_LSTM = "demand_lstm"
    SUPPLIER_XGB = "supplier_xgb"
    INVENTORY_DQN = "inventory_dqn"
    PROCUREMENT_OPT = "procurement_opt"


class ModelStatus(StrEnum):
    TRAINED = "trained"
    VALIDATED = "validated"
    STAGING = "staging"
    PRODUCTION = "production"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class TrainingJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --- Recommendations ------------------------------------------------------
class RecommendationStatus(StrEnum):
    GENERATED = "generated"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"
    ORDER_CREATED = "order_created"
    EXPIRED = "expired"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ScenarioType(StrEnum):
    DEMAND_SPIKE = "demand_spike"
    LEAD_TIME_INCREASE = "lead_time_increase"
    SUPPLIER_FAILURE = "supplier_failure"
    SUPPLIER_CAPACITY_REDUCTION = "supplier_capacity_reduction"
    INVENTORY_LEVEL = "inventory_level"
    BUDGET_CONSTRAINT = "budget_constraint"
    WAREHOUSE_CAPACITY = "warehouse_capacity"
    PRICE_CHANGE = "price_change"
    PROMOTION = "promotion"
    SERVICE_TARGET = "service_target"
    CUSTOM = "custom"