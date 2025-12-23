"""causal_data_gen: lightweight synthetic SCM generator for continuous variables.

Main entrypoint: `CausalSCM.random(...)` and `scm.sample(n)`.
"""

from .scm import CausalSCM, SCMMetadata
from .dag import RandomDAGConfig
from .datasets import Dataset, generate_dataset

__all__ = [
    "CausalSCM",
    "SCMMetadata",
    "RandomDAGConfig",
    "Dataset",
    "generate_dataset",
]
