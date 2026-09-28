"""Feature engineering layer.

Public API:
    from data_platform.features import build_all_features, FEATURE_VERSION
"""

from data_platform.features.feature_registry import FEATURE_VERSION, FeatureSpec, REGISTRY
from data_platform.features.feature_store import build_all_features, load_feature_table

__all__ = [
    "FEATURE_VERSION",
    "FeatureSpec",
    "REGISTRY",
    "build_all_features",
    "load_feature_table",
]