"""Concrete repositories for each ORM aggregate."""

from db.repositories.base_repository import BaseRepository
from db.repositories.category_repository import CategoryRepository
from db.repositories.demand_repository import DemandRepository
from db.repositories.inventory_repository import InventoryRepository
from db.repositories.model_repository import ModelRepository
from db.repositories.order_repository import PurchaseOrderRepository
from db.repositories.product_repository import ProductRepository
from db.repositories.recommendation_repository import RecommendationRepository
from db.repositories.shipment_repository import ShipmentRepository
from db.repositories.supplier_product_repository import SupplierProductRepository
from db.repositories.supplier_repository import SupplierRepository
from db.repositories.user_repository import UserRepository
from db.repositories.warehouse_repository import WarehouseRepository

__all__ = [
    "BaseRepository",
    "CategoryRepository",
    "DemandRepository",
    "InventoryRepository",
    "ModelRepository",
    "PurchaseOrderRepository",
    "ProductRepository",
    "RecommendationRepository",
    "ShipmentRepository",
    "SupplierProductRepository",
    "SupplierRepository",
    "UserRepository",
    "WarehouseRepository",
]