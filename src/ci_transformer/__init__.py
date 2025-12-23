"""Unified namespace for causal data generation, CI models, and training utilities."""

from .causal_data_gen import (
    CausalSCM,
    SCMMetadata,
    RandomDAGConfig,
    Dataset,
    generate_dataset,
)
from .ci_models import CISetTransformer, CISetTransformerConfig
from .training import CIStreamingDataset, Curriculum, make_dataloader, train_ci_model

__all__ = [
    "CausalSCM",
    "SCMMetadata",
    "RandomDAGConfig",
    "Dataset",
    "generate_dataset",
    "CISetTransformer",
    "CISetTransformerConfig",
    "CIStreamingDataset",
    "Curriculum",
    "make_dataloader",
    "train_ci_model",
]

