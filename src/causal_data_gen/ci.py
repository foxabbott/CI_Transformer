from __future__ import annotations
from collections import deque
from typing import Iterable, List, Set, Tuple
import numpy as np

Array = np.ndarray

def parents(adj: Array, v: int) -> List[int]:
    return np.where(adj[:, v] != 0)[0].astype(int).tolist()

def children(adj: Array, v: int) -> List[int]:
    return np.where(adj[v, :] != 0)[0].astype(int).tolist()

def descendants(adj: Array, v: int) -> Set[int]:
    # DFS
    stack = [v]
    seen = set([v])
    out = set()
    while stack:
        u = stack.pop()
        for c in children(adj, u):
            if c not in seen:
                seen.add(c)
                out.add(c)
                stack.append(c)
    out.discard(v)
    return out

def d_separated(adj: Array, X: Iterable[int], Y: Iterable[int], Z: Iterable[int]) -> bool:
    """Bayes-ball d-separation test for DAGs.

    Returns True if X and Y are d-separated given Z.
    Assumes `adj[i,j]=1` denotes edge i->j and adj is acyclic.

    Reference: Koller & Friedman (Bayes-ball algorithm).
    """
    X = set(map(int, X))
    Y = set(map(int, Y))
    Z = set(map(int, Z))

    # Reverse reachability from Z to mark nodes that have a descendant in Z:
    # A node has a descendant in Z iff it can reach some z by following edges forward.
    # So we can mark all ancestors of Z by traversing parents from each z.
    has_desc_in_Z = np.zeros(adj.shape[0], dtype=bool)
    stack = list(Z)
    seen = set(stack)
    while stack:
        v = stack.pop()
        for p in parents(adj, v):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    for v in seen:
        has_desc_in_Z[v] = True

    # Bayes-ball state: (node, direction) where direction in {"up","down"}
    # "up": coming from a child; "down": coming from a parent.
    q = deque()
    visited = set()

    for x in X:
        q.append((x, "up"))
        q.append((x, "down"))

    while q:
        v, direction = q.popleft()
        if (v, direction) in visited:
            continue
        visited.add((v, direction))

        if v in Y:
            return False  # active path found

        if v in Z:
            # observed node blocks some flows
            if direction == "up":
                # can go up to parents only
                for p in parents(adj, v):
                    q.append((p, "up"))
            # if direction == "down": stop
        else:
            # unobserved
            if direction == "up":
                # go up to parents and down to children
                for p in parents(adj, v):
                    q.append((p, "up"))
                for c in children(adj, v):
                    q.append((c, "down"))
            else:  # direction == "down"
                # if v is a collider? In bayes-ball, collider behavior depends on direction:
                # Coming from parent into v (down) means we arrived via an arrow into v.
                # We can go down to children always.
                for c in children(adj, v):
                    q.append((c, "down"))
                # and we can go up to parents if v has descendant in Z (collider opened)
                if has_desc_in_Z[v]:
                    for p in parents(adj, v):
                        q.append((p, "up"))

    return True
