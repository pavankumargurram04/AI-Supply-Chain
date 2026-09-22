"""Import every ORM model so SQLAlchemy's mapper registry sees the full graph.

Order matters only for readability — SQLAlchemy resolves `relationship()`
strings lazily at configure time.
"""

from db.base import Base
from db.models.alert import Alert
from db.models.audit_log import AuditLog
from db.models.category import Category
from db.models.demand import DemandRecord
from db.models.inventory import Inventory
from db.models.inventory_movement import InventoryMovement
from db.models.model_registry import ModelRegistry
from db.models.order import PurchaseOrder, PurchaseOrderItem
from db.models.prediction import Prediction
from db.models.product import Product
from db.models.quality_event import QualityEvent
from db.models.recommendation import RecommendationItem, RecommendationRun
from db.models.return_ import Return
from db.models.scenario import ScenarioRun
from db.models.shipment import Shipment
from db.models.shipment_event import ShipmentEvent
from db.models.stockout import Stockout
from db.models.supplier import Supplier
from db.models.supplier_event import SupplierEvent
from db.models.supplier_product import SupplierProduct
from db.models.training_job import TrainingJob
from db.models.user import User
from db.models.warehouse import Warehouse

__all__ = [
    "Base",
    "Alert",
    "AuditLog",
    "Category",
    "DemandRecord",
    "Inventory",
    "InventoryMovement",
    "ModelRegistry",
    "PurchaseOrder",
    "PurchaseOrderItem",
    "Prediction",
    "Product",
    "QualityEvent",
    "RecommendationItem",
    "RecommendationRun",
    "Return",
    "ScenarioRun",
    "Shipment",
    "ShipmentEvent",
    "Stockout",
    "Supplier",
    "SupplierEvent",
    "SupplierProduct",
    "TrainingJob",
    "User",
    "Warehouse",
]