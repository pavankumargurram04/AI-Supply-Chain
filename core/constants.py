"""Project-wide constants.

Values here are static (not tunable). Tunable parameters live in
`core.config.Settings`.
"""

from __future__ import annotations

# --- Versioning of artefacts -----------------------------------------------
DATA_VERSION_FORMAT = "%Y%m%d_%H%M%S"
FEATURE_VERSION_FORMAT = "%Y%m%d_%H%M%S"
MODEL_VERSION_PREFIX = "v"

# --- Forecasting / inventory defaults -------------------------------------
DEFAULT_FORECAST_HORIZON_DAYS = 30
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_MONTE_CARLO_SAMPLES = 10_000
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_LEAD_TIME_STD_DAYS = 1.5

# --- Cost defaults (only defaults — real values come from DB) -------------
DEFAULT_HOLDING_COST_RATE = 0.20        # fraction of unit cost / year
DEFAULT_ORDER_FIXED_COST = 250.0        # per purchase order
DEFAULT_SHORTAGE_COST_MULTIPLIER = 3.0  # multiple of unit cost

# --- Pagination ------------------------------------------------------------
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# --- Time ------------------------------------------------------------------
DAY_SECONDS = 86_400

# --- Deterministic salt strings --------------------------------------------
SEED_DEMAND = "demand"
SEED_SUPPLIER = "supplier"
SEED_INVENTORY = "inventory"
SEED_AGENTS = "agents"

# --- Audit action names ----------------------------------------------------
AUDIT_CREATE = "create"
AUDIT_UPDATE = "update"
AUDIT_DELETE = "delete"
AUDIT_APPROVE = "approve"
AUDIT_REJECT = "reject"
AUDIT_OVERRIDE = "override"
AUDIT_PROMOTE = "promote"
AUDIT_ROLLBACK = "rollback"
AUDIT_TRAIN = "train"
AUDIT_RUN_DECISION = "run_decision"
AUDIT_RUN_SCENARIO = "run_scenario"