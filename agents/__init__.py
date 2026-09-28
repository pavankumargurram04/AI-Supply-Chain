"""Multi-agent orchestration.

Agents exchange structured `AgentMessage` objects. Each agent wraps a
specific step of the decision engine and emits a typed message that becomes
part of the trace. No agent invents numbers; each agent calls the actual
model or engine function.

Public API:
    from agents import AgentOrchestrator

    orch = AgentOrchestrator(
        demand_checkpoint_dir="artifacts/demand/dev",
        supplier_checkpoint_dir="artifacts/supplier/dev",
        inventory_checkpoint_dir="artifacts/inventory/dev",
        curated_features_dir="datasets/curated/dev/features",
    )
    result = orch.run(product_sku="SKU-000000", warehouse_code="WH-NY")
"""

from agents.coordinator_agent import AgentOrchestrator, OrchestratorResult
from agents.messages import AgentMessage, AgentRole

__all__ = ["AgentOrchestrator", "OrchestratorResult", "AgentMessage", "AgentRole"]