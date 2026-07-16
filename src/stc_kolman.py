import math
import time
from dataclasses import dataclass
from typing import Set, Dict, List, Hashable
import networkx as nx

from src.stc_arv import arv_balanced_cut
from src.stc_core import (canon_edge, validate_tree, compute_tree_congestion, CongestionResult,
                          _ensure_small_enough, _DSU, _relabel_graph_to_ints, _compute_cut_sizes)

Node = Hashable

@dataclass
class ApproxSTCResult:
    tree: nx.Graph
    congestion: CongestionResult
    solve_time_seconds: float

MAX_BALANCED_CUT_N = 20 # variable controlls dispatcher branch triggering, <=val does exact cut computation else arv
def dispatcher(H: nx.Graph) -> Set[Node]:
    """
    Balanced-cut oracle with two branches by subgraph size:
    - n <= MAX_BALANCED_CUT_N: exact minimum 2/3-balanced cut (alpha = 1)
    - larger: ARV SDP relaxation + simplified rounding (see stc_arv.py)
    """
    if H.number_of_nodes() <= MAX_BALANCED_CUT_N:
        return _exact_balanced_cut_simple(H) #exact exponential
    return arv_balanced_cut(H)

def _exact_balanced_cut_simple(H: nx.Graph) -> Set[Node]:
    r"""
    mask: bitset encoding one candidate side S of the cut
    - bit v = 1 => vertex v in S,
    - else 0 and is in V\S
    mask = 0b10110 => S={1,2,4}

    computes minimum 2/3-balanced cut by full subset search 2^(n-1)- could be done better i suppose...
    returns one side S of the cut, minimizes cut_size[mask]
    """
    #checking basics
    if not nx.is_connected(H): raise ValueError("H must be connected.")
    G2, _, back = _relabel_graph_to_ints(H)
    n = G2.number_of_nodes()
    _ensure_small_enough(n, MAX_BALANCED_CUT_N, "_exact_balanced_cut_simple")
    if n <= 1: return set(H.nodes())

    cut_size, _, popcount = _compute_cut_sizes(G2)

    lower, upper = math.ceil(n / 3), math.floor(2 * n / 3)
    best_mask, best_value = None, None

    for mask in range(1, 2**n, 2): #by symmetry every second mask is duplicate
        size = popcount[mask]
        if size < lower or size > upper:
            continue
        value = cut_size[mask]
        if best_value is None or value < best_value:
            best_value = value
            best_mask = mask
            if value == 0: break

    if best_mask is None:
        best_mask = 1

    answer = set()
    for v in range(n):
        if best_mask & (1 << v):
            answer.add(back[v])
    return answer


def kolman_main(G: nx.Graph) -> ApproxSTCResult:
    r"""
    Kolman's CongSpanTree recursive construction (paper Section 3, Algorithm 1):
    - partition the graph by a 2/3-balanced cut (via dispatcher: exact for
      n <= 20, ARV-based above),
    - remove cut edges, recurse on the resulting connected components,
    - reconnect the component spanning trees using cut edges (union-find).

    Why this works (paper's analysis):
    - hereditary bisection width lower-bounds STC: STC(G) >= Omega(hb(G)/maxdeg),
    - each recursion level adds at most the cut width to the congestion,
    - balanced cuts make the recursion depth O(log n),
    - so congestion <= O(alpha(n) log n) * hb(G), and with the lower bound
      this gives the O(alpha(n) log n * maxdeg) approximation ratio.

    Note: our ARV branch uses a simplified rounding (see stc_arv.py), so the
    proven alpha(n) = O(sqrt(log n)) attaches to the full ARV rounding, not
    to ours; the structural inequality congestion <= O(log n) * max cut width
    holds for our implementation unconditionally.
    """
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")

    start = time.perf_counter()

    #recursive divide-and-conquer on components
    def _recurse(H: nx.Graph) -> nx.Graph:
        """8 lines of pseudo code from kolmans paper turned into networkx implementation. Section 3"""
        if H.number_of_nodes() == 1:
            T = nx.Graph()
            T.add_nodes_from(H.nodes())
            return T

        if H.number_of_edges() == H.number_of_nodes() - 1: #tree
            return H.copy()

        side = dispatcher(H)

        # guard if the oracle returns cut in [ceil(n/3), floor(2n/3)]
        _n = H.number_of_nodes()
        _a = len(side)
        _b = _n - _a
        _lo, _hi = math.ceil(_n / 3), math.floor(2 * _n / 3)
        if not (_lo <= _a <= _hi and _lo <= _b <= _hi):
            raise RuntimeError(
                f"oracle returned a non-2/3-balanced cut at n={_n}: "
                f"sides {_a}/{_b}, allowed [{_lo},{_hi}]")

        cut_edges = []
        for u, v in H.edges():
            if (u in side) != (v in side):
                cut_edges.append(canon_edge(u, v))

        H_without_cut = H.copy()
        H_without_cut.remove_edges_from(cut_edges)


        components = [set(comp) for comp in nx.connected_components(H_without_cut)]
        comp_index: Dict[Node, int] = {}
        trees: List[nx.Graph] = []
        for idx, comp in enumerate(components):
            for u in comp:
                comp_index[u] = idx
            sub = H.subgraph(comp).copy()
            trees.append(_recurse(sub))

        T = nx.Graph()
        T.add_nodes_from(H.nodes())
        for sub_tree in trees:
            T.add_edges_from(sub_tree.edges())

        dsu = _DSU(range(len(components)))
        for u, v in cut_edges:
            cu = comp_index[u]
            cv = comp_index[v]
            if dsu.union(cu, cv):
                T.add_edge(u, v)

        if T.number_of_edges() != H.number_of_nodes() - 1 or not nx.is_tree(T):
            raise RuntimeError("Recursive construction did not return a spanning tree.")
        return T

    T = _recurse(G.copy())
    validate_tree(G, T)
    congestion = compute_tree_congestion(G, T)
    return ApproxSTCResult(T, congestion, time.perf_counter() - start)
