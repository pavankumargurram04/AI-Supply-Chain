"""Inventory monitor.

Checks recent stockouts, service level, and stockout alert counts against
reference windows. Uses the `stockouts` and `inventory_movements` tables
when the DB session is passed; otherwise reads Parquet snapshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.logging import get_logger
from monitoring.alerting import MonitorReport

log = get_logger(__name__)


def monitor_inventory(
    *,
    movements_path: Optional[str | Path] = None,
    recent_days: int = 30,
    reference_days: int = 180,
    stockout_alert_threshold: int = 50,
) -> MonitorReport:
    """Return a MonitorReport for inventory operations.

    `stockout_alert_threshold` is a *monitoring* threshold, not a business
    decision rule: crossing it emits an alert for a human to inspect.
    """
    report = MonitorReport(name="inventory_monitor")

    if movements_path is None:
        default_path = Path("datasets/raw")
        # Look for the newest inventory_movements parquet under any mode.
        candidates = []
        for sub in default_path.glob("*/inventory_movements"):
            p = sub / "data.parquet"
            if p.exists():
                candidates.append(p)
        if not candidates:
            report.notes.append("no inventory_movements parquet found")
            return report
        movements_path = max(candidates, key=lambda p: p.stat().st_mtime)

    df = pd.read_parquet(movements_path)
    if df.empty:
        report.notes.append("empty inventory movements")
        return report

    df["occurred_at"] = pd.to_datetime(df["occurred_at"])
    max_date = df["occurred_at"].max()
    recent_start = max_date - pd.Timedelta(days=recent_days)
    reference_start = max_date - pd.Timedelta(days=recent_days + reference_days)

    recent = df[df["occurred_at"] > recent_start]
    reference = df[(df["occurred_at"] > reference_start) & (df["occurred_at"] <= recent_start)]

    recent_stockouts = int((recent["movement_type"] == "stockout").sum())
    ref_stockouts = int((reference["movement_type"] == "stockout").sum())

    # Normalise per day for a fair comparison.
    recent_daily = recent_stockouts / max(1, recent_days)
    ref_daily = ref_stockouts / max(1, reference_days)

    report.metrics["recent_stockout_count"] = recent_stockouts
    report.metrics["reference_stockout_count"] = ref_stockouts
    report.metrics["recent_stockout_per_day"] = recent_daily
    report.metrics["reference_stockout_per_day"] = ref_daily
    report.metrics["stockout_rate_change"] = (
        (recent_daily - ref_daily) / ref_daily if ref_daily > 0 else float("nan")
    )

    # Issue movements as a proxy for demand served.
    recent_issues = int((recent["movement_type"] == "issue").sum())
    report.metrics["recent_issue_count"] = recent_issues
    report.metrics["stockout_to_issue_ratio"] = (
        recent_stockouts / max(recent_issues, 1)
    )

    if recent_stockouts > stockout_alert_threshold:
        report.add_alert(
            source="inventory_stockouts",
            severity="warning",
            title=f"{recent_stockouts} stockouts in the last {recent_days} days",
            message=(
                f"threshold {stockout_alert_threshold} exceeded; "
                f"recent rate {recent_daily:.2f}/day vs reference {ref_daily:.2f}/day"
            ),
            entity_type="inventory",
            payload={
                "recent_stockouts": recent_stockouts,
                "reference_stockouts": ref_stockouts,
            },
        )

    if ref_daily > 0 and recent_daily > 1.5 * ref_daily:
        report.add_alert(
            source="inventory_stockout_ratio",
            severity="warning",
            title="Recent stockout rate materially above reference window",
            message=(
                f"{recent_daily:.3f}/day vs reference {ref_daily:.3f}/day "
                f"(+{100*(recent_daily/ref_daily - 1):.1f}%)"
            ),
            entity_type="inventory",
            payload={"recent_daily": recent_daily, "reference_daily": ref_daily},
        )

    return report