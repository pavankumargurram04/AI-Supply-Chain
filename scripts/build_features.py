"""CLI: build feature tables from a curated dataset.

Usage:
    python scripts/build_features.py --mode dev
    python scripts/build_features.py --input datasets/raw/research --output datasets/curated/research
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging import configure_logging, get_logger  # noqa: E402
from data_platform.features.feature_store import build_all_features  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build feature tables.")
    parser.add_argument("--mode", choices=["dev", "research", "stress"], default=None)
    parser.add_argument("--input", type=str, default=None,
                        help="Input dataset directory (contains Parquet tables).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for feature Parquet files.")
    args = parser.parse_args()

    configure_logging()

    if args.input is None and args.mode is None:
        parser.error("Provide --input or --mode.")
    input_dir = args.input or f"datasets/raw/{args.mode}"
    output_dir = args.output or f"datasets/curated/{args.mode or Path(input_dir).name}"

    if not Path(input_dir).exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    result = build_all_features(input_dir, output_dir, mode=args.mode)

    print(f"Feature version: {result.feature_version}")
    print(f"Output directory: {result.output_dir}")
    print("Row counts:")
    for k, v in result.row_counts.items():
        print(f"  {k:24s} {v:>12,d}")
    print(f"Total: {result.row_counts and sum(result.row_counts.values()):,d}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
