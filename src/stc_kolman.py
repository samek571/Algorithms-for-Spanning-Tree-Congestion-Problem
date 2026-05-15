import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Optional, Dict, List, Hashable
import networkx as nx

from src.stc_arv import arv_balanced_cut
from src.stc_core import (canon_edge, choose_root_by_degree, validate_tree, compute_tree_congestion, CongestionResult,
                          _ensure_small_enough, _DSU, _relabel_graph_to_ints, _compute_cut_sizes)

Node = Hashable

@dataclass
class ApproxSTCResult:
    tree: nx.Graph
    congestion: CongestionResult
    solve_time_seconds: float

MAX_BALANCED_CUT_N = 20


def dispatcher(H: nx.Graph) -> Set[Node]:
    """ hybrid cut oracle"""
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
    exact cut oracle replacing the paper’s polynomial-time approximation oracle,

    It doesnt implement thm1 from paper as it is not using balanced cut approximation
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

#helper in approx cut finder
def _best_two_thirds_balanced_prefix_cut(H: nx.Graph, order: list[Node]) -> Optional[Set[Node]]:
    """
    Given an ordering of the vertices, test all prefix sets and return the
    2/3-balanced prefix with minimum cut size

    If order = [v1, v2, ..., vn], then the candidates are:
        {v1}, {v1,v2}, ..., {v1,...,v_{n-1}}.

    prefixes has to lie in [ceil(n/3), floor(2n/3)]
    """
    n = H.number_of_nodes()
    if n <= 1: return set(H.nodes())

    lower, upper = math.ceil(n / 3), math.floor(2 * n / 3)

    all_nodes = set(order)
    side, best_side, best_value = set(), None, None

    for i, u in enumerate(order[:-1], start=1):
        side.add(u)
        if i < lower or i > upper: continue #cut guarantee

        value = nx.cut_size(H, side, all_nodes - side)
        if best_value is None or value < best_value:
            best_value = value
            best_side = set(side)

    return best_side


def _approx_balanced_cut_simple(H: nx.Graph) -> Set[Node]:
    """
    Polynomial-time 2/3-balanced cut heuristic.

    This is NOT a literal implementation of Theorem 1 from Kolman's paper.
    Theorem 1 invokes external black-box algorithms (ARV / KPR), whereas
    this provides practical substitute for this codebase

    Strategy:
    1. Try a spectral sweep cut:
       - compute the Fiedler vector of the normalized Laplacian,
       - sort vertices by the vector values,
       - evaluate all 2/3-balanced prefixes,
       - return the one with smallest cut size
    2. Fallback to Kernighan-Lin bisection, guaranteed 2/3
    3. Fallback to the BFS-layer prefix heuristic.
    4. Final deterministic fallback: first ceil(n/3) vertices.
    """
    if not nx.is_connected(H):
        raise ValueError("H must be connected.")

    n = H.number_of_nodes()
    if n <= 1:
        return set(H.nodes())

    nodes = list(H.nodes())

    # 1) Spectral sweep cut
    try:
        import numpy as np
        from scipy.sparse.linalg import eigsh

        L = nx.normalized_laplacian_matrix(H, nodelist=nodes).astype(float) # graph connectivity in LA form
        #if 2 verts are tightly tied inside the same region of the graph, the spectral embedding tends to place them near each other,
        _, vecs = eigsh(L, k=2, which="SM") #asks for 2 eigenvectors by SmallestMAgnitude
        #for connected graph smallest ev = 0, second one is Fiedler vector - supposedly used to guess cuts?!
        fiedler = np.asarray(vecs[:, 1]).reshape(-1) #taking only the second ev per vertex

        order = [u for _, u in sorted(zip(fiedler, nodes), key=lambda x: x[0])] #sort by fiedler value
        side = _best_two_thirds_balanced_prefix_cut(H, order) #for allowed 1/3 prefix it comuputes cut size and returns best one
        if side is not None:
            return side

        # also try reverse order in case the better side is suffix
        side = _best_two_thirds_balanced_prefix_cut(H, list(reversed(order)))
        if side is not None:
            return side
    except Exception: # it all is in try catch block as a lot of things might go wrong, i dont really undestand LinearAlgebra on that deep level idk if there always is second ev...
        pass

    # 2) Kernighan-Lin fallback
    try:
        left, right = nx.community.kernighan_lin_bisection(H)
        if set(left) and set(right):
            return left
    except Exception:
        pass

    # 3) Existing BFS-prefix fallback
    root = choose_root_by_degree(H)
    side = _bfs_layer_prefix_cut(H, root=root, min_balance=1/3)
    if side is not None and 0 < len(side) < n:
        return side

    # 4) Deterministic last resort
    lower = math.ceil(n / 3)
    return set(nodes[:lower])

#coarse helper in the fallback branch approx/bruteforce
def _bfs_layer_prefix_cut(G: nx.Graph, root: Node, min_balance: float = 0.25) -> Optional[set[Node]]:
    r"""
    subheuristic if 2/3 cut fails, just a cheap guess of a cut

    BFS yields layers from root, we try to find a connected side A by taking a prefix such accomodation
    A0, A0uA1, A0uA1uA2 ...

    among prefixes that satisfy min balance we choose such minimizing cut_size / min(|A|, |B|)
    returned side is connected by construction
    Kolmanss preliminaries define edge expansion as Beta(G) = min_A [e(A,V\A) / min(|A|, |V\A|)]
    """

    levels = nx.single_source_shortest_path_length(G, root)
    by_level: Dict[int, List[Node]] = defaultdict(list)
    for u, d in levels.items():
        by_level[d].append(u)

    all_levels = sorted(by_level)
    if len(all_levels) <= 1:
        return None

    n = G.number_of_nodes()
    seen = set()
    best_A: Optional[set[Node]] = None
    best_score: Optional[float] = None

    for depth in all_levels[:-1]:
        seen.update(by_level[depth])
        a = len(seen)
        b = n - a
        if a == 0 or b == 0:
            continue
        if min(a, b) < max(1, int(min_balance * n)):
            continue
        cut_size = nx.cut_size(G, seen, set(G.nodes()) - seen)
        score = cut_size / min(a, b)
        if best_score is None or score < best_score:
            best_score = score
            best_A = set(seen)

    return best_A


def kolman_main(G: nx.Graph) -> ApproxSTCResult:
    r"""
    Kolmans-style recursive construction, but with an exact exponential balanced-cut subroutine for clarity purposes
    - hereditary bisection width gives a lower bound on STC,
    - recursive balanced cuts keep adding only the cut-size overhead per recursion level,
    - because cuts are balanced, recursion depth is O(log n),
    - so total congestion scales like O(alpha(n) log n) * hb(G),
    - and with the lower bound STC(G) = Ω(hb(G)/Δ), that becomes the approximation ratio.

    then recursive balanced partitioning should help construct low-congestion trees.
    (STC(G)>=\omega(hb(G)/maxdeg)) where hb is hereditary bisection width

    - partition the graph by a 2/3-balanced cut,
    - remove cut edges and solve recursively on the resulting connected components,
    - combine the component spanning trees into a spanning tree of the whole graph from cut set

    tree constructor-first, cng analysis-second, deviation form paper as we dont implement the oracle explicitly
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
