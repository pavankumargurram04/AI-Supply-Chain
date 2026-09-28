from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_platform.features.supplier_features import (
    build_supplier_features_memory_safe,
)

FEATURE_VERSION = "v1_96d567ed8f88"

rows, products = build_supplier_features_memory_safe(
    "datasets/raw/research",
    "datasets/curated/research",
    feature_version=FEATURE_VERSION,
)

print()
print("SUPPLIER FEATURE BUILD COMPLETE")
print(f"Products: {products:,}")
print(f"Rows:     {rows:,}")
print(
    "Output:   "
    f"datasets/curated/research/features/"
    f"supplier_features_{FEATURE_VERSION}.parquet"
)
