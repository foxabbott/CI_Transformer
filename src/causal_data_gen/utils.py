from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

Array = np.ndarray

def rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)

def softclip(x: Array, lo: float = -10.0, hi: float = 10.0) -> Array:
    # Prevent explosions in exotic mechanisms while keeping differentiability.
    return np.tanh((x - lo) / (hi - lo) * 2 - 1) * (hi - lo) / 2 + (hi + lo) / 2

def standardize(x: Array, eps: float = 1e-8) -> Array:
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    return (x - m) / (s + eps)

def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))

def safe_exp(x: Array, cap: float = 8.0) -> Array:
    return np.exp(np.clip(x, -cap, cap))

def choose(rng: np.random.Generator, items: Sequence[Any], p: Optional[Sequence[float]] = None):
    idx = rng.choice(len(items), p=p)
    return items[int(idx)]

def topological_sort(adj: Array) -> list[int]:
    """Kahn's algorithm. Assumes adj is a DAG adjacency matrix (d x d), adj[i,j]=1 means i->j."""
    d = adj.shape[0]
    indeg = adj.sum(axis=0).astype(int).tolist()
    q = [i for i in range(d) if indeg[i] == 0]
    order = []
    while q:
        i = q.pop()
        order.append(i)
        children = np.where(adj[i] != 0)[0]
        for j in children:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(int(j))
    if len(order) != d:
        raise ValueError("Adjacency matrix is cyclic; cannot topo-sort.")
    return order
