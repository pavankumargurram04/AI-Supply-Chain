"""Generator configuration and mode presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable, Dict


@dataclass
class GeneratorConfig:
    """Everything needed to reproduce a dataset run."""

    mode: str
    seed: int
    n_categories: int
    n_products: int
    n_suppliers: int
    n_warehouses: int
    start_date: date
    end_date: date
    output_dir: str

    # Feature toggles — locked into the manifest so experiments are honest.
    enable_seasonality: bool = True
    enable_promotion: bool = True
    enable_shocks: bool = True
    enable_supplier_events: bool = True
    enable_inventory_ledger: bool = True

    # --- Derived ---
    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        d["days"] = self.days
        return d

    # --- Presets ---
    @classmethod
    def dev(cls, seed: int = 42, output_dir: str = "datasets/raw/dev") -> "GeneratorConfig":
        """Tiny dataset for pipeline debugging. Sub-minute generation."""
        return cls(
            mode="dev",
            seed=seed,
            n_categories=5,
            n_products=30,
            n_suppliers=8,
            n_warehouses=3,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            output_dir=output_dir,
        )

    @classmethod
    def research(cls, seed: int = 42, output_dir: str = "datasets/raw/research") -> "GeneratorConfig":
        """Research dataset — meets the spec minimum (250+ products, 40+ suppliers, 10+ warehouses, 5 years)."""
        return cls(
            mode="research",
            seed=seed,
            n_categories=20,
            n_products=250,
            n_suppliers=40,
            n_warehouses=10,
            start_date=date(2019, 1, 1),
            end_date=date(2023, 12, 31),
            output_dir=output_dir,
        )

    @classmethod
    def stress(cls, seed: int = 42, output_dir: str = "datasets/raw/stress") -> "GeneratorConfig":
        """Stress dataset — performance and data-engineering tests."""
        return cls(
            mode="stress",
            seed=seed,
            n_categories=40,
            n_products=1000,
            n_suppliers=100,
            n_warehouses=20,
            start_date=date(2019, 1, 1),
            end_date=date(2023, 12, 31),
            output_dir=output_dir,
        )


MODE_PRESETS: Dict[str, Callable[..., GeneratorConfig]] = {
    "dev": GeneratorConfig.dev,
    "research": GeneratorConfig.research,
    "stress": GeneratorConfig.stress,
}