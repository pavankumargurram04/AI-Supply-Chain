"""Aggregate all validation checks for a dataset directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from data_platform.cleaning.demand_cleaner import clean_demand
from data_platform.cleaning.inventory_cleaner import clean_inventory
from data_platform.cleaning.supplier_cleaner import clean_supplier_events
from data_platform.ingestion.parquet_writer import read_table
from data_platform.quality.reports import QualityReport
from data_platform.validation.duplicate_detector import KEYS, detect_duplicates
from data_platform.validation.missingness_report import missingness
from data_platform.validation.range_validator import RULES, validate_ranges
from data_platform.validation.schema_validator import SCHEMAS, validate_schema


def _load_all(input_dir: str) -> Dict[str, pd.DataFrame]:
    tables = [
        "categories", "products", "suppliers", "warehouses",
        "supplier_products", "supplier_events",
        "demand_records", "inventory", "inventory_movements",
    ]
    out: Dict[str, pd.DataFrame] = {}
    for t in tables:
        df = read_table(input_dir, t)
        out[t] = df
    return out


def run_quality_checks(input_dir: str) -> QualityReport:
    tables = _load_all(input_dir)
    report = QualityReport(input_dir=input_dir)

    for name, df in tables.items():
        report.row_counts[name] = int(len(df))
        if df.empty:
            report.notes.append(f"{name}: empty")
            continue

        # Schema
        schema = SCHEMAS.get(name)
        if schema is not None:
            issues = validate_schema(df, schema)
            for issue in issues:
                report.schema_issues.append(issue.to_dict())

        # Ranges
        rules = RULES.get(name, [])
        if rules:
            range_issues = validate_ranges(df, name, rules)
            for issue in range_issues:
                report.range_issues.append(issue.to_dict())

        # Duplicates
        key_cols = KEYS.get(name, [])
        if key_cols:
            dup = detect_duplicates(df, name, key_cols)
            report.duplicate_reports.append(dup.to_dict())

        # Missingness
        miss = missingness(df, name)
        report.missingness.append(miss.to_dict())

    # Cleaning pass (only for the tables that have cleaners).
    if not tables["demand_records"].empty:
        _, sum_d = clean_demand(tables["demand_records"])
        report.clean_summaries.append(sum_d.to_dict())

    if not tables["supplier_events"].empty:
        _, sum_s = clean_supplier_events(
            tables["supplier_events"],
            known_supplier_codes=tables["suppliers"]["code"].tolist(),
        )
        report.clean_summaries.append(sum_s.to_dict())

    if not tables["inventory"].empty or not tables["inventory_movements"].empty:
        _, _, sum_i = clean_inventory(
            tables["inventory"], tables["inventory_movements"]
        )
        report.clean_summaries.append(sum_i.to_dict())

    return report