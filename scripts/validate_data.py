"""CLI: run the full data-quality pipeline against a generated dataset.

Usage:
    python scripts/validate_data.py --input datasets/raw/dev
    python scripts/validate_data.py --mode dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging import configure_logging, get_logger  # noqa: E402
from data_platform.quality.checks import run_quality_checks  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a synthetic dataset.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--mode", choices=["dev", "research", "stress"], default=None)
    parser.add_argument("--report", type=str, default=None,
                        help="Path to write the JSON report. Defaults to reports/metrics/quality_<mode>.json")
    args = parser.parse_args()

    configure_logging()

    if args.input is None and args.mode is None:
        parser.error("Provide --input or --mode.")
    input_dir = args.input or f"datasets/raw/{args.mode}"

    if not Path(input_dir).exists():
        print(f"Input directory does not exist: {input_dir}")
        return 1

    log.info("validate.start", extra={"input": input_dir})
    report = run_quality_checks(input_dir)

    report_path = args.report or f"reports/metrics/quality_{Path(input_dir).name}.json"
    report.write(report_path)

    print("Row counts:")
    for k, v in report.row_counts.items():
        print(f"  {k:24s} {v:>12,d}")
    print()
    print(f"Schema issues:       {len(report.schema_issues)}")
    print(f"Range issues:        {len(report.range_issues)}")
    print(f"Duplicate reports:   {len(report.duplicate_reports)}")
    for rep in report.duplicate_reports:
        if rep.get("duplicate_rows", 0) > 0:
            print(f"  - {rep['table']}: {rep['duplicate_rows']} duplicate rows on {rep['key_columns']}")
    print(f"Missingness reports: {len(report.missingness)}")
    print(f"Cleaning summaries:  {len(report.clean_summaries)}")
    print(f"\nPASSED: {report.passed}")
    print(f"Report written to:   {report_path}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())