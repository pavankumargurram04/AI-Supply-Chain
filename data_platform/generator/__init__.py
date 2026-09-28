"""Deterministic enterprise-grade synthetic data generator.

Public entry point:

    from data_platform.generator import generate_all, GeneratorConfig
    cfg = GeneratorConfig.research()
    tables = generate_all(cfg)
"""

from data_platform.generator.config import GeneratorConfig, MODE_PRESETS

__all__ = ["GeneratorConfig", "MODE_PRESETS"]