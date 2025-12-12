"""Utilities to train a conditional-independence classifier on causal_synth.

- Streaming IterableDataset that samples random SCMs and CI queries
- Curriculum scheduling over conditioning-set size m
"""
from .streaming import CIStreamingDataset, Curriculum, make_dataloader
from .train import train_ci_model

__all__ = ["CIStreamingDataset", "Curriculum", "make_dataloader", "train_ci_model"]
