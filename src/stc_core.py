
from __future__ import annotations

import math
from collections import deque
from dataclasses import  dataclass
from typing import Callable, Dict, Hashable, List, Optional, Tuple
import networkx as nx

Node = Hashable
Edge = Tuple[Node, Node]
TreeBuilder = Callable[[nx.Graph], nx.Graph]

@dataclass
class CongestionResult:
    edge_congestion: Dict[Edge, int]
    max_congestion: int


#helper
def canon_edge(u: Node, v: Node) -> Edge:
    """Canonical undir edge representation, so we are consistent"""
    return tuple(sorted((u, v), key=repr))  # type: ignore[return-value]

#helper
def choose_root_by_degree(G: nx.Graph) -> Node:
    return max(G.nodes(), key=lambda x: (G.degree[x], repr(x)))

#helper
def validate_tree(G: nx.Graph, T: nx.Graph) -> None:
    """Maximal connected acyclic tree on induced edges using all vertices = spanning tree"""
    if set(G.nodes()) != set(T.nodes()):
        raise ValueError("T must use exactly the same vertex set as G.")
    if T.number_of_edges() != G.number_of_nodes() - 1:
        raise ValueError("T must have exactly n-1 edges.")
    if not nx.is_tree(T):
        raise ValueError("T must be a connected acyclic graph.")
    for u, v in T.edges():
        if not G.has_edge(u, v):
            raise ValueError(f"Tree edge {(u, v)} does not belong to G.")

#helper
def _prepare_rooted_tree(T: nx.Graph, root: Node):
    parent: Dict[Node, Optional[Node]] = {root: None}
    depth = {root: 0}
    order = []

    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in T.neighbors(u):
            if v == parent[u]:
                continue
            if v in parent:
                continue
            parent[v] = u
            depth[v] = depth[u] + 1
            q.append(v)

    if len(order) != T.number_of_nodes():
        raise ValueError("T is not connected.")

    n = len(order)
    lg = max(1, n.bit_length())
    up: Dict[Node, List[Optional[Node]]] = {u: [None] * lg for u in T.nodes()}
    for u in T.nodes():
        up[u][0] = parent[u]
    for j in range(1, lg):
        for u in T.nodes():
            mid = up[u][j - 1]
            up[u][j] = None if mid is None else up[mid][j - 1]

    return parent, depth, order, up, lg

#helper
def _lca(u: Node, v: Node, depth, up, lg: int) -> Node:
    if depth[u] < depth[v]:
        u, v = v, u

    diff = depth[u] - depth[v]
    bit = 0
    while diff:
        if diff & 1:
            u = up[u][bit]
        diff >>= 1
        bit += 1

    if u == v:
        return u

    for j in range(lg - 1, -1, -1):
        if up[u][j] != up[v][j]:
            u = up[u][j]
            v = up[v][j]

    return up[u][0]


def compute_tree_congestion(G: nx.Graph, T: nx.Graph, root: Optional[Node] = None) -> CongestionResult:
    """
    Computes congestion of every tree edge in O(m log n) time.

    For each graph edge {u,v} in G, add +1 to every tree edge on the unique u-v path in T,
    hence automatically contributes of tree edges themselves
    """
    validate_tree(G, T)
    if root is None:
        root = next(iter(T.nodes()))

    parent, depth, order, up, lg = _prepare_rooted_tree(T, root)
    delta = {u:0 for u in T.nodes()}
    for u, v in G.edges():
        w = _lca(u, v, depth, up, lg)
        delta[u] += 1
        delta[v] += 1
        delta[w] -= 2

    edge_congestion: Dict[Edge, int] = {}
    max_congestion = 0
    for u in reversed(order):
        p = parent[u]
        if p is None:
            continue
        c = delta[u]
        edge_congestion[canon_edge(u, p)] = c
        max_congestion = max(max_congestion, c)
        delta[p] += c

    return CongestionResult(edge_congestion=edge_congestion, max_congestion=max_congestion)


def graph_statistics(G: nx.Graph) -> Dict[str, float]:
    """Lightweight graph descriptors for CSV analysis."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    degrees = [deg for _, deg in G.degree()]

    if n == 0:
        return {
            "n": 0,
            "m": 0,
            "min_degree": 0,
            "max_degree": 0,
            "avg_degree": 0.0,
            "degree_variance": 0.0,
            "degree_stddev": 0.0,
            "density": 0.0,
            "n_articulation_points": 0,
            "n_bridges": 0,
            "n_leaves": 0,
            "is_regular": 1,
        }

    avg_degree = 2 * m / n
    if len(degrees) == 1:
        degree_variance = 0.0
    else:
        degree_variance = sum((d - avg_degree) ** 2 for d in degrees) / n
    degree_stddev = math.sqrt(degree_variance)
    density = nx.density(G)

    try:
        n_articulation_points = sum(1 for _ in nx.articulation_points(G)) if nx.is_connected(G) else 0
    except Exception:
        n_articulation_points = 0
    try:
        n_bridges = sum(1 for _ in nx.bridges(G)) if nx.is_connected(G) else 0
    except Exception:
        n_bridges = 0

    return {
        "n": n,
        "m": m,
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "avg_degree": avg_degree,
        "degree_variance": degree_variance,
        "degree_stddev": degree_stddev,
        "density": density,
        "n_articulation_points": n_articulation_points,
        "n_bridges": n_bridges,
        "n_leaves": sum(1 for d in degrees if d == 1),
        "is_regular": int(min(degrees) == max(degrees)),
    }

