"""CLI: package an experiment run directory as a zip with a reproducibility README.

Usage:
    python scripts\export_experiment.py --run-dir experiments/runs/exp_dev_seed42_20240101_120000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging import configure_logging  # noqa: E402
from experiments.report import build_report  # noqa: E402


README_TEMPLATE = """# Experiment: {experiment_id}

Generated: {created_at}
Mode: {mode}
Seed: {seed}
Git commit: {git_commit}
Host: {hostname}

## Versions

- Feature version: `{feature_version}`
- Data version: `{data_version}`
- Demand model version: `{demand_model_version}`
- Supplier model version: `{supplier_model_version}`
- Inventory model version: `{inventory_model_version}`

## Contents

- `tables/` — CSV and JSON metric tables (real values only)
- `plots/` — figures referenced by the manuscript
- `paper/` — Markdown tables ready for the write-up
- `manifest.json` — full provenance record
- `summary.json` — aggregate results

## Reproducing

1. Install dependencies: `pip install -r requirements.txt`
2. Generate and ingest the dataset for mode `{mode}`: