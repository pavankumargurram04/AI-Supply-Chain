"""Global demand forecasting model (LSTM with embeddings).

Public entry points:
    from ml.demand.dataset import DemandSequenceBuilder, WindowedDemandDataset
    from ml.demand.model import GlobalDemandLSTM
    from ml.demand.train import train_demand_model
"""

from ml.demand.dataset import (
    DemandDatasetBundle,
    DemandScalers,
    WindowedDemandDataset,
    build_demand_datasets,
)
from ml.demand.model import GlobalDemandLSTM, DemandModelConfig

__all__ = [
    "DemandDatasetBundle",
    "DemandScalers",
    "WindowedDemandDataset",
    "build_demand_datasets",
    "GlobalDemandLSTM",
    "DemandModelConfig",
]