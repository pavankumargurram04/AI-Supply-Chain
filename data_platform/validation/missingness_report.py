"""Per-column missingness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class MissingnessReport:
    table: str
    rows: int
    columns: Dict[str, dict]  # col -> {missing, pct_missing, dtype}

    @property
    def total_cells(self) -> int:
        return self.rows * len(self.columns)

    @property
    def total_missing(self) -> int:
        return sum(v["missing"] for v in self.columns.values())

    @property
    def overall_pct(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return 100.0 * self.total_missing / self.total_cells

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "rows": self.rows,
            "overall_pct_missing": round(self.overall_pct, 4),
            "columns": self.columns,
        }


def missingness(df: pd.DataFrame, table_name: str) -> MissingnessReport:
    n = len(df)
    cols: Dict[str, dict] = {}
    for col in df.columns:
        missing = int(df[col].isna().sum())
        cols[col] = {
            "missing": missing,
            "pct_missing": round(100.0 * missing / n, 4) if n else 0.0,
            "dtype": str(df[col].dtype),
        }
    return MissingnessReport(table=table_name, rows=n, columns=cols)