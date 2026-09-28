"""Master data generators: categories, products, suppliers, warehouses."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import numpy as np

from core.enums import (
    LifecycleStage,
    ReliabilityClass,
    SeasonalityClass,
    VolatilityClass,
)
from data_platform.generator.config import GeneratorConfig


# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------

# Base list; extended systematically if more are needed.
_BASE_CATEGORIES: List[tuple[str, str]] = [
    ("ELEC", "Electronics"),
    ("APPL", "Home Appliances"),
    ("TOOL", "Tools & Hardware"),
    ("CLTH", "Apparel"),
    ("FOOD", "Packaged Foods"),
    ("BEVG", "Beverages"),
    ("HOME", "Home & Garden"),
    ("SPRT", "Sports & Outdoors"),
    ("TOYS", "Toys & Games"),
    ("AUTO", "Automotive"),
    ("HLTH", "Health & Beauty"),
    ("OFFC", "Office Supplies"),
    ("PET",  "Pet Supplies"),
    ("BABY", "Baby Products"),
    ("INDL", "Industrial"),
    ("SAFE", "Safety & Security"),
    ("STAT", "Stationery"),
    ("FURN", "Furniture"),
    ("GROC", "Groceries"),
    ("TEXT", "Textiles"),
]

_SUBCATEGORY_SUFFIXES = ["Basic", "Premium", "Bulk", "Seasonal", "Pro", "Eco"]


def _expand_category_names(n: int) -> List[tuple[str, str]]:
    """Return exactly `n` unique (code, name) category pairs."""
    out: List[tuple[str, str]] = []
    i = 0
    while len(out) < n:
        if i < len(_BASE_CATEGORIES):
            out.append(_BASE_CATEGORIES[i])
        else:
            suffix = _SUBCATEGORY_SUFFIXES[i % len(_SUBCATEGORY_SUFFIXES)]
            idx = i // len(_SUBCATEGORY_SUFFIXES)
            base_code, base_name = _BASE_CATEGORIES[idx % len(_BASE_CATEGORIES)]
            code = f"{base_code}-{suffix[:2].upper()}{i}"
            out.append((code, f"{base_name} — {suffix}"))
        i += 1
    return out


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

def generate_categories(cfg: GeneratorConfig, rng: np.random.Generator) -> List[Dict[str, Any]]:
    names = _expand_category_names(cfg.n_categories)
    rows: List[Dict[str, Any]] = []
    for code, name in names:
        rows.append({
            "code": code,
            "name": name,
            "description": f"Synthetic category {name}",
            "parent_code": None,
        })
    return rows


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

# Category-level demand scale, cost midpoint, and typical lead time.
_CATEGORY_PROFILE: Dict[str, Dict[str, float]] = {
    "ELEC": {"demand_scale": 1.4, "cost_mid": 45.0,  "lead": 21},
    "APPL": {"demand_scale": 0.9, "cost_mid": 120.0, "lead": 28},
    "TOOL": {"demand_scale": 1.1, "cost_mid": 18.0,  "lead": 14},
    "CLTH": {"demand_scale": 2.0, "cost_mid": 12.0,  "lead": 21},
    "FOOD": {"demand_scale": 3.2, "cost_mid": 4.0,   "lead": 7},
    "BEVG": {"demand_scale": 4.0, "cost_mid": 2.5,   "lead": 5},
    "HOME": {"demand_scale": 1.3, "cost_mid": 22.0,  "lead": 18},
    "SPRT": {"demand_scale": 1.0, "cost_mid": 30.0,  "lead": 21},
    "TOYS": {"demand_scale": 1.8, "cost_mid": 14.0,  "lead": 25},
    "AUTO": {"demand_scale": 0.8, "cost_mid": 60.0,  "lead": 30},
    "HLTH": {"demand_scale": 2.2, "cost_mid": 9.0,   "lead": 12},
    "OFFC": {"demand_scale": 2.5, "cost_mid": 6.0,   "lead": 10},
    "PET":  {"demand_scale": 2.8, "cost_mid": 11.0,  "lead": 12},
    "BABY": {"demand_scale": 2.0, "cost_mid": 15.0,  "lead": 18},
    "INDL": {"demand_scale": 0.7, "cost_mid": 80.0,  "lead": 30},
    "SAFE": {"demand_scale": 0.9, "cost_mid": 35.0,  "lead": 21},
    "STAT": {"demand_scale": 3.5, "cost_mid": 3.5,   "lead": 7},
    "FURN": {"demand_scale": 0.5, "cost_mid": 150.0, "lead": 35},
    "GROC": {"demand_scale": 4.5, "cost_mid": 3.0,   "lead": 5},
    "TEXT": {"demand_scale": 1.6, "cost_mid": 20.0,  "lead": 21},
}
_DEFAULT_PROFILE = {"demand_scale": 1.0, "cost_mid": 20.0, "lead": 14}


def _profile_for(code: str) -> Dict[str, float]:
    return _CATEGORY_PROFILE.get(code, _DEFAULT_PROFILE)


_LIFECYCLE_CHOICES = (
    [LifecycleStage.MATURITY] * 6
    + [LifecycleStage.GROWTH] * 2
    + [LifecycleStage.INTRODUCTION] * 1
    + [LifecycleStage.DECLINE] * 1
)
_SEASONALITY_CHOICES = (
    [SeasonalityClass.NONE] * 3
    + [SeasonalityClass.LOW] * 3
    + [SeasonalityClass.MEDIUM] * 3
    + [SeasonalityClass.HIGH] * 2
    + [SeasonalityClass.VERY_HIGH] * 1
)
_VOLATILITY_CHOICES = (
    [VolatilityClass.LOW] * 3
    + [VolatilityClass.MEDIUM] * 4
    + [VolatilityClass.HIGH] * 2
    + [VolatilityClass.EXTREME] * 1
)


def generate_products(
    cfg: GeneratorConfig,
    categories: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cat_codes = [c["code"] for c in categories]

    for i in range(cfg.n_products):
        cat = cat_codes[i % len(cat_codes)]
        prof = _profile_for(cat.split("-")[0])  # normalise sub-codes

        unit_cost = float(np.clip(rng.lognormal(mean=np.log(prof["cost_mid"]), sigma=0.55), 0.5, 5000.0))
        margin = float(rng.uniform(1.30, 2.60))
        unit_price = round(unit_cost * margin, 4)
        unit_cost = round(unit_cost, 4)

        weight_kg = float(np.clip(rng.lognormal(mean=np.log(0.6), sigma=1.1), 0.005, 500.0))
        volume_m3 = float(np.clip(weight_kg / 250.0 * rng.uniform(0.5, 2.0), 1e-5, 5.0))

        base_demand = float(np.clip(rng.lognormal(mean=np.log(8.0), sigma=1.0) * prof["demand_scale"], 0.5, 5000.0))

        rows.append({
            "sku": f"SKU-{i:06d}",
            "name": f"{cat} Item {i:04d}",
            "description": f"Synthetic product in category {cat}",
            "category_code": cat,
            "unit_cost": unit_cost,
            "unit_price": unit_price,
            "weight_kg": round(weight_kg, 4),
            "volume_m3": round(volume_m3, 6),
            "lifecycle_stage": str(rng.choice(_LIFECYCLE_CHOICES)),
            "seasonality_class": str(rng.choice(_SEASONALITY_CHOICES)),
            "volatility_class": str(rng.choice(_VOLATILITY_CHOICES)),
            "base_demand": round(base_demand, 4),
            "lead_time_days": int(prof["lead"]) + int(rng.integers(-3, 4)),
        })
    return rows


# ---------------------------------------------------------------------------
# Suppliers
# ---------------------------------------------------------------------------

_REGION_POOL = [
    ("APAC", "CN"), ("APAC", "IN"), ("APAC", "VN"),
    ("EMEA", "DE"), ("EMEA", "TR"), ("EMEA", "PL"),
    ("AMER", "US"), ("AMER", "MX"), ("AMER", "BR"),
]

_RELIABILITY_CHOICES = (
    [ReliabilityClass.PREMIUM] * 15
    + [ReliabilityClass.RELIABLE] * 35
    + [ReliabilityClass.STANDARD] * 35
    + [ReliabilityClass.POOR] * 15
)


def generate_suppliers(cfg: GeneratorConfig, rng: np.random.Generator) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(cfg.n_suppliers):
        region, country = _REGION_POOL[i % len(_REGION_POOL)]
        reliability = str(rng.choice(_RELIABILITY_CHOICES))
        cost_class = str(rng.choice(["low", "standard", "premium"], p=[0.35, 0.5, 0.15]))
        quality_class = str(rng.choice(["low", "standard", "high"], p=[0.2, 0.55, 0.25]))
        lead_time_class = str(rng.choice(["short", "standard", "long"], p=[0.3, 0.5, 0.2]))
        capacity_class = str(rng.choice(["small", "medium", "large"], p=[0.35, 0.5, 0.15]))

        cap_mult = {"small": 5_000, "medium": 50_000, "large": 500_000}[capacity_class]
        capacity = float(rng.integers(cap_mult // 2, cap_mult * 2))

        rows.append({
            "code": f"SUP-{i:04d}",
            "name": f"Supplier {i:04d}",
            "contact_email": f"contact{i:04d}@supplier-{i:04d}.example",
            "contact_phone": f"+1-555-{i:04d}",
            "country": country,
            "region": region,
            "reliability_class": reliability,
            "cost_class": cost_class,
            "quality_class": quality_class,
            "lead_time_class": lead_time_class,
            "capacity_class": capacity_class,
            "contracted_capacity_units": capacity,
            "notes": f"Synthetic supplier #{i:04d}",
        })
    return rows


# ---------------------------------------------------------------------------
# Warehouses
# ---------------------------------------------------------------------------

_WAREHOUSE_LOCATIONS = [
    ("WH-NY", "New York DC",   "AMER", "US", 40.7128, -74.0060),
    ("WH-LA", "Los Angeles DC","AMER", "US", 34.0522, -118.2437),
    ("WH-CHI","Chicago DC",    "AMER", "US", 41.8781, -87.6298),
    ("WH-TOR","Toronto DC",    "AMER", "CA", 43.6532, -79.3832),
    ("WH-MEX","Mexico City DC","AMER", "MX", 19.4326, -99.1332),
    ("WH-LON","London DC",     "EMEA", "GB", 51.5074, -0.1278),
    ("WH-BER","Berlin DC",     "EMEA", "DE", 52.5200, 13.4050),
    ("WH-PAR","Paris DC",      "EMEA", "FR", 48.8566,  2.3522),
    ("WH-WAW","Warsaw DC",     "EMEA", "PL", 52.2297, 21.0122),
    ("WH-DXB","Dubai DC",      "EMEA", "AE", 25.2048, 55.2708),
    ("WH-BLR","Bengaluru DC",  "APAC", "IN", 12.9716, 77.5946),
    ("WH-SHA","Shanghai DC",   "APAC", "CN", 31.2304, 121.4737),
    ("WH-SIN","Singapore DC",  "APAC", "SG",  1.3521, 103.8198),
    ("WH-TOK","Tokyo DC",      "APAC", "JP", 35.6762, 139.6503),
    ("WH-SYD","Sydney DC",     "APAC", "AU", -33.8688, 151.2093),
    ("WH-SAO","Sao Paulo DC",  "AMER", "BR", -23.5505, -46.6333),
    ("WH-JNB","Johannesburg DC","EMEA","ZA", -26.2041, 28.0473),
    ("WH-IST","Istanbul DC",   "EMEA", "TR", 41.0082, 28.9784),
    ("WH-BKK","Bangkok DC",    "APAC", "TH", 13.7563, 100.5018),
    ("WH-CAI","Cairo DC",      "EMEA", "EG", 30.0444, 31.2357),
]


def generate_warehouses(cfg: GeneratorConfig, rng: np.random.Generator) -> List[Dict[str, Any]]:
    if cfg.n_warehouses > len(_WAREHOUSE_LOCATIONS):
        raise ValueError(
            f"Need {cfg.n_warehouses} warehouses but only {len(_WAREHOUSE_LOCATIONS)} locations defined."
        )
    rows: List[Dict[str, Any]] = []
    for code, name, region, country, lat, lon in _WAREHOUSE_LOCATIONS[: cfg.n_warehouses]:
        capacity = float(rng.integers(100_000, 2_000_000))
        op_cost = float(rng.integers(2_000, 15_000))
        rows.append({
            "code": code,
            "name": name,
            "region": region,
            "country": country,
            "latitude": float(lat),
            "longitude": float(lon),
            "capacity_units": capacity,
            "operating_cost_daily": op_cost,
        })
    return rows