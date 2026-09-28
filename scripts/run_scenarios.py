"""CLI: run a batch of scenarios across a set of (product, warehouse) pairs.

Usage:
    python scripts\run_scenarios.py --mode dev --scenarios demand_spike_50,inventory_500 --series SKU-000000:WH-NY
    python scripts\run_scenarios.py --mode dev --all --series-file series.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging import configure_logging, get_logger  # noqa: E402
from scenario_engine import SCENARIOS, run_batch  # noqa: E402

log = get_logger(__name__)


def _parse_series(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"Expected SKU:WAREHOUSE, got {s!r}")
    sku, wh = s.split(":", 1)
    return sku, wh


def main() -> int:
    p = argparse.ArgumentParser(description="Run scenarios in batch.")
    p.add_argument("--mode", choices=["dev", "research", "stress"], default="dev")
    p.add_argument("--scenarios", type=str, default=None,
                   help="Comma-separated scenario keys.")
    p.add_argument("--all", action="store_true", help="Run every scenario in the catalog.")
    p.add_argument("--series", type=_parse_series, nargs="+", default=None,
                   help="One or more SKU:WAREHOUSE pairs.")
    p.add_argument("--series-file", type=str, default=None,
                   help="Path to a file with one SKU:WAREHOUSE per line.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-persist", action="store_true")
    p.add_argument("--output", type=str, default="reports/scenarios")
    args = p.parse_args()

    configure_logging()

    # Resolve scenario keys.
    if args.all:
        keys = list(SCENARIOS.keys())
    elif args.scenarios:
        keys = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        p.error("Provide either --all or --scenarios")

    unknown = [k for k in keys if k not in SCENARIOS]
    if unknown:
        p.error(f"Unknown scenarios: {unknown}")

    # Resolve series.
    series: list[tuple[str, str]] = list(args.series or [])
    if args.series_file:
        for line in Path(args.series_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                series.append(_parse_series(line))
    if not series:
        p.error("Provide at least one --series or a --series-file")

    results = run_batch(
        scenario_keys=keys,
        series=series,
        mode=args.mode,
        seed=args.seed,
        output_dir=args.output,
        persist=not args.no_persist,
    )

    print(f"Ran {len(results)} scenario(s) across {len(series)} series.")
    print(f"JSON: {args.output}\\scenarios_{args.mode}.json")
    print(f"CSV:  {args.output}\\scenarios_{args.mode}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())