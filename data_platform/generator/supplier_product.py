"""Synthetic supplier-product relationship generator."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from data_platform.generator.config import GeneratorConfig


def generate_supplier_products(
    cfg: GeneratorConfig,
    products: List[Dict[str, Any]],
    suppliers: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Generate realistic supplier-product commercial relationships.

    The generator layer uses stable business identifiers rather than
    database-generated integer primary keys.

    Output identifiers:
        product_sku
        supplier_code

    Commercial fields:
        unit_price
        moq
        max_order_qty
        lead_time_days
        lead_time_std_days
        shipping_cost
        is_preferred
    """

    if not products:
        return []

    if not suppliers:
        raise ValueError("At least one supplier is required.")

    supplier_count = len(suppliers)

    if cfg.mode == "dev":
        min_links = min(2, supplier_count)
        max_links = min(4, supplier_count)
    elif cfg.mode == "research":
        min_links = min(3, supplier_count)
        max_links = min(8, supplier_count)
    else:
        min_links = min(4, supplier_count)
        max_links = min(10, supplier_count)

    cost_multiplier = {
        "low": 0.92,
        "standard": 1.00,
        "premium": 1.12,
    }

    lead_multiplier = {
        "short": 0.75,
        "standard": 1.00,
        "long": 1.35,
    }

    quality_multiplier = {
        "low": 1.10,
        "standard": 1.00,
        "high": 0.96,
    }

    reliability_multiplier = {
        "premium": 0.95,
        "reliable": 0.98,
        "standard": 1.00,
        "poor": 1.06,
    }

    supplier_link_counts = {
        supplier["code"]: 0
        for supplier in suppliers
    }

    rows: List[Dict[str, Any]] = []

    for product in products:
        n_links = int(rng.integers(min_links, max_links + 1))

        # Prefer suppliers with fewer existing relationships while
        # retaining deterministic randomness.
        weights = np.array(
            [
                1.0 / (1.0 + supplier_link_counts[s["code"]])
                for s in suppliers
            ],
            dtype=float,
        )
        weights /= weights.sum()

        selected_indices = rng.choice(
            supplier_count,
            size=n_links,
            replace=False,
            p=weights,
        )

        candidates: List[Dict[str, Any]] = []

        for supplier_index in selected_indices:
            supplier = suppliers[int(supplier_index)]

            supplier_cost = cost_multiplier.get(
                supplier["cost_class"], 1.0
            )
            supplier_quality = quality_multiplier.get(
                supplier["quality_class"], 1.0
            )
            supplier_reliability = reliability_multiplier.get(
                supplier["reliability_class"], 1.0
            )
            supplier_lead = lead_multiplier.get(
                supplier["lead_time_class"], 1.0
            )

            product_cost = float(product["unit_cost"])
            product_price = float(product["unit_price"])
            product_lead = float(product["lead_time_days"])

            # Supplier procurement price is influenced by supplier
            # commercial/quality/reliability characteristics.
            price_factor = (
                supplier_cost
                * supplier_quality
                * supplier_reliability
                * float(rng.uniform(0.92, 1.08))
            )

            unit_price = product_cost * price_factor

            # Keep procurement price below downstream selling price.
            unit_price = min(unit_price, product_price * 0.90)
            unit_price = max(unit_price, product_cost * 0.70)

            # MOQ scales with expected demand.
            base_moq = max(
                1,
                int(np.ceil(float(product["base_demand"]) * 2.0)),
            )

            moq = max(
                1,
                int(
                    np.ceil(
                        base_moq * float(rng.uniform(0.5, 2.0)) / 5.0
                    )
                    * 5
                ),
            )

            contracted_capacity = float(
                supplier.get("contracted_capacity_units") or 100_000
            )

            max_order_qty = int(
                max(
                    moq,
                    min(
                        contracted_capacity
                        * float(rng.uniform(0.05, 0.25)),
                        float(product["base_demand"])
                        * float(rng.uniform(30.0, 120.0)),
                    ),
                )
            )

            # Lead time combines product characteristics and
            # supplier-specific lead-time class.
            lead_time = (
                product_lead
                * supplier_lead
                * float(rng.uniform(0.85, 1.15))
            )

            lead_time = float(
                np.clip(lead_time, 1.0, 180.0)
            )

            base_std = max(1.0, lead_time * 0.12)

            reliability_std_factor = {
                "premium": 0.65,
                "reliable": 0.80,
                "standard": 1.00,
                "poor": 1.45,
            }.get(
                supplier["reliability_class"],
                1.0,
            )

            lead_time_std = float(
                np.clip(
                    base_std
                    * reliability_std_factor
                    * float(rng.uniform(0.8, 1.2)),
                    0.25,
                    lead_time * 0.60,
                )
            )

            weight = float(product.get("weight_kg") or 0.1)
            volume = float(product.get("volume_m3") or 0.001)

            shipping_cost = (
                1.50
                + weight * float(rng.uniform(0.8, 2.2))
                + volume * float(rng.uniform(20.0, 80.0))
                + lead_time * float(rng.uniform(0.05, 0.20))
            )

            shipping_cost = float(
                max(0.50, shipping_cost)
            )

            candidates.append(
                {
                    "product_sku": product["sku"],
                    "supplier_code": supplier["code"],
                    "unit_price": round(unit_price, 4),
                    "moq": moq,
                    "max_order_qty": max_order_qty,
                    "lead_time_days": round(lead_time, 2),
                    "lead_time_std_days": round(lead_time_std, 2),
                    "shipping_cost": round(shipping_cost, 4),
                    "is_preferred": False,
                    "_score": (
                        unit_price
                        + shipping_cost
                        + lead_time * 0.5
                        + lead_time_std * 0.75
                    ),
                }
            )

            supplier_link_counts[supplier["code"]] += 1

        # Select the commercially strongest candidate as preferred.
        candidates.sort(key=lambda row: row["_score"])

        for rank, row in enumerate(candidates):
            row["is_preferred"] = rank == 0
            row.pop("_score", None)
            rows.append(row)

    return rows
