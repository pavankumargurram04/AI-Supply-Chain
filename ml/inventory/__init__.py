"""Inventory intelligence subsystem.

Delivers:
    - Monte Carlo stockout probability
    - Safety stock derived from service level
    - Reorder point from expected lead-time demand
    - Cost model
    - DQN environment consuming real forecasts and supplier state

Public entry points are re-exported here for convenience.
"""

from ml.inventory.cost_model import (
    CostParameters,
    holding_cost,
    ordering_cost,
    shortage_cost,
    total_cost,
)
from ml.inventory.reorder_point import reorder_point_from_simulation
from ml.inventory.safety_stock import (
    safety_stock_from_simulation,
    safety_stock_normal_approx,
)
from ml.inventory.simulation import (
    LeadTimeDemandResult,
    SimulationConfig,
    simulate_lead_time_demand,
)

__all__ = [
    "CostParameters",
    "holding_cost",
    "ordering_cost",
    "shortage_cost",
    "total_cost",
    "LeadTimeDemandResult",
    "SimulationConfig",
    "simulate_lead_time_demand",
    "safety_stock_from_simulation",
    "safety_stock_normal_approx",
    "reorder_point_from_simulation",
]