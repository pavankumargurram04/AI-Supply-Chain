"""Partitioned Parquet writer.

Writes every table under `<output_dir>/<table>/...` and produces a
`_manifest.json` describing the run.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.logging import get_logger

log = get_logger(__name__)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_table(
    df: pd.DataFrame,
    output_dir: str,
    table_name: str,
    *,
    partition_by: Optional[list[str]] = None,
) -> Path:
    """Write a single table to Parquet. Returns the destination path."""
    base = Path(output_dir) / table_name
    _ensure_dir(base)

    if df.empty:
        # Still create an empty file so downstream code can rely on existence.
        target = base / "part-00000.parquet"
        df.to_parquet(target, index=False)
        log.info("parquet.written", extra={"table": table_name, "rows": 0, "path": str(target)})
        return base

    if partition_by:
        # Replace existing partitioned output to keep generation idempotent.
        if base.exists():
            shutil.rmtree(base)
        _ensure_dir(base)

        # Partitioned write (e.g. year/month for demand).
        df.to_parquet(base, index=False, partition_cols=partition_by)
        rows = len(df)
        log.info("parquet.written.partitioned",
                 extra={"table": table_name, "rows": rows, "keys": partition_by})
    else:
        target = base / "data.parquet"
        df.to_parquet(target, index=False)
        log.info("parquet.written", extra={"table": table_name, "rows": len(df), "path": str(target)})
    return base


def write_manifest(
    output_dir: str,
    *,
    mode: str,
    seed: int,
    config: Dict[str, Any],
    tables: Dict[str, int],
) -> Path:
    """Write a JSON manifest describing a generation run."""
    _ensure_dir(Path(output_dir))
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "mode": mode,
        "seed": seed,
        "config": config,
        "table_row_counts": tables,
        "total_rows": sum(tables.values()),
    }
    target = Path(output_dir) / "_manifest.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("manifest.written", extra={"path": str(target), "tables": len(tables)})
    return target


def read_table(output_dir: str, table_name: str) -> pd.DataFrame:
    base = Path(output_dir) / table_name
    if not base.exists():
        return pd.DataFrame()
    return pd.read_parquet(base)

