"""Sierpinski gasket graph generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _SierpinskiGraph:
    adj: dict[int, set[int]]
    corners: tuple[int, int, int]


def _base_graph() -> _SierpinskiGraph:
    adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    return _SierpinskiGraph(adj=adj, corners=(0, 1, 2))


def _copy_graph(graph: _SierpinskiGraph, offset: int) -> _SierpinskiGraph:
    new_adj: dict[int, set[int]] = {}
    for node, nbrs in graph.adj.items():
        new_adj[node + offset] = {nbr + offset for nbr in nbrs}
    corners = tuple(c + offset for c in graph.corners)
    return _SierpinskiGraph(adj=new_adj, corners=corners)  # type: ignore[arg-type]


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def _compress_graph(adj: dict[int, set[int]], parent: list[int]) -> dict[int, set[int]]:
    mapping: dict[int, int] = {}
    next_id = 0
    for node in adj.keys():
        root = _find_root(parent, node)
        if root not in mapping:
            mapping[root] = next_id
            next_id += 1
    new_adj: dict[int, set[int]] = {mapping[_find_root(parent, node)]: set() for node in adj.keys()}
    for node, nbrs in adj.items():
        a = mapping[_find_root(parent, node)]
        for nbr in nbrs:
            b = mapping[_find_root(parent, nbr)]
            if a != b:
                new_adj[a].add(b)
                new_adj[b].add(a)
    return new_adj


def _find_root(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _build_level(level: int) -> _SierpinskiGraph:
    if level == 0:
        return _base_graph()

    prev = _build_level(level - 1)
    n_prev = len(prev.adj)
    top = _copy_graph(prev, 0)
    left = _copy_graph(prev, n_prev)
    right = _copy_graph(prev, 2 * n_prev)

    total_nodes = 3 * n_prev
    parent, find, union = _union_find(total_nodes)

    # Merge corners as specified.
    union(top.corners[1], left.corners[0])
    union(top.corners[2], right.corners[0])
    union(left.corners[2], right.corners[1])

    merged_adj: dict[int, set[int]] = {}
    for graph in (top, left, right):
        for node, nbrs in graph.adj.items():
            merged_adj.setdefault(node, set()).update(nbrs)

    compressed_adj = _compress_graph(merged_adj, parent)

    top_corner = _find_root(parent, top.corners[0])
    left_corner = _find_root(parent, left.corners[1])
    right_corner = _find_root(parent, right.corners[2])

    mapping: dict[int, int] = {}
    next_id = 0
    for node in merged_adj.keys():
        root = _find_root(parent, node)
        if root not in mapping:
            mapping[root] = next_id
            next_id += 1

    corners = (mapping[top_corner], mapping[left_corner], mapping[right_corner])
    return _SierpinskiGraph(adj=compressed_adj, corners=corners)


def sierpinski(level: int, *, lazy: float = 0.0) -> tuple[dict[int, set[int]], np.ndarray]:
    """Build Sierpinski gasket graph at recursion level.

    Returns (adjacency_dict, P) where P is the lazy random walk kernel on the graph.
    """
    if level < 0:
        raise ValueError("level must be nonnegative")
    if lazy < 0.0 or lazy >= 1.0:
        raise ValueError("lazy must be in [0, 1)")

    graph = _build_level(level)
    adj = graph.adj
    n = len(adj)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        neighbors = sorted(adj[i])
        deg = len(neighbors)
        if deg == 0:
            P[i, i] = 1.0
            continue
        P[i, i] = lazy
        share = (1.0 - lazy) / float(deg)
        for j in neighbors:
            P[i, j] = share
    return adj, P
