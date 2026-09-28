"""Decision engine: orchestrates demand, supplier, inventory into a
reproducible, explainable procurement recommendation.

Public API:
    from decision_engine import DecisionEngine, DecisionRequest

    engine = DecisionEngine(
        demand_checkpoint_dir="artifacts/demand/dev",
        supplier_checkpoint_dir="artifacts/supplier/dev",
        inventory_checkpoint_dir="artifacts/inventory/dev",
        curated_features_dir="datasets/curated/dev/features",
    )
    result = engine.run(DecisionRequest(product_sku="SKU-0001", warehouse_code="WH-NY"))
"""

from decision_engine.decision_trace import DecisionTrace, DecisionRequest
from decision_engine.engine import DecisionEngine

__all__ = ["DecisionEngine", "DecisionRequest", "DecisionTrace"]