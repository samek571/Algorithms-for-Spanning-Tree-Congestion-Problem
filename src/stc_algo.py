from __future__ import annotations

import heapq
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple

import networkx as nx

from stc_core import CongestionResult, canon_edge, compute_tree_congestion, validate_tree, choose_root_by_degree

Node = Hashable
Edge = Tuple[Node, Node]

MAX_EXACT_N = 20 #2^number so lets stay realistic within bounds of the atoms in universe (more than 20 is pretty bad)
MAX_BALANCED_CUT_N = 22


@dataclass
class ExactSTCResult:
    optimum_congestion: int
    tree: nx.Graph
    solve_time_seconds: float


@dataclass
class ApproxSTCResult:
    tree: nx.Graph
    congestion: CongestionResult
    solve_time_seconds: float

#helper
def _ensure_small_enough(n: int, limit: int, name: str) -> None:
    if n > limit:
        raise ValueError(
            f"{name} run in exponential time complexity, try lower, ideally n <= {limit}."
            f"Received n={n}."
        )

#helper
def _relabel_graph_to_ints(G: nx.Graph) -> tuple[nx.Graph, Dict[Node, int], Dict[int, Node]]:
    old_nodes = list(G.nodes())
    to_int = {}
    to_old = {}
    for i, node in enumerate(old_nodes):
        to_int[node] = i
        to_old[i] = node
    H = nx.Graph()
    H.add_nodes_from(range(len(old_nodes)))
    for u, v in G.edges():
        H.add_edge(to_int[u], to_int[v])
    return H, to_int, to_old

#helper
def _popcount(x: int) -> int:
    return x.bit_count()

#helper
def _all_popcounts(fullmask: int) -> List[int]:
    pop = [0] * (fullmask + 1)
    for mask in range(1, fullmask + 1):
        pop[mask] = pop[mask >> 1] + (mask & 1)
    return pop

#helper lemma 3.1+3.2 basically caching so we dont compute cuts exhaustively
def _compute_cut_sizes(H: nx.Graph) -> tuple[List[int], List[int], List[int]]:
    n = H.number_of_nodes()
    fullmask = (1 << n) - 1

    degree = [0] * n
    neigh_mask = [0] * n
    for v in range(n):
        degree[v] = H.degree[v]
    for u, v in H.edges():
        neigh_mask[u] |= 1 << v
        neigh_mask[v] |= 1 << u

    cut_size = [0] * (fullmask + 1)
    for mask in range(1, fullmask + 1):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        inside_neighbors = _popcount(neigh_mask[v] & rest)
        cut_size[mask] = cut_size[rest] + degree[v] - 2 * inside_neighbors

    popcount = _all_popcounts(fullmask)
    return cut_size, neigh_mask, popcount


#helper 
class _DSU:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {}
        self.rank = {}
        for x in items:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


#tree builder
def bfs_tree(G: nx.Graph, root: Optional[Node] = None) -> nx.Graph:
    if root is None:
        root = choose_root_by_degree(G)
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    T.add_edges_from(nx.bfs_edges(G, root))
    return T

#tree builder
def dfs_tree(G: nx.Graph, root: Optional[Node] = None) -> nx.Graph:
    if root is None:
        root = choose_root_by_degree(G)
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    T.add_edges_from(nx.dfs_edges(G, root))
    return T


def uniform_spanning_tree(G: nx.Graph, seed: Optional[int] = None, root: Optional[Node] = None) -> nx.Graph:
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")

    rng = random.Random(seed)
    nodes = list(G.nodes())
    if root is None:
        root = rng.choice(nodes)

    T = nx.Graph()
    T.add_nodes_from(nodes)
    in_tree = {root}

    while len(in_tree) < len(nodes):
        start = rng.choice([u for u in nodes if u not in in_tree])
        path = [start]
        pos = {start: 0}
        u = start

        while u not in in_tree:
            v = rng.choice(list(G.neighbors(u)))
            if v in pos:
                idx = pos[v]
                for rem in path[idx + 1 :]:
                    pos.pop(rem, None)
                path = path[: idx + 1]
            else:
                path.append(v)
                pos[v] = len(path) - 1
            u = v

        for a, b in zip(path, path[1:]):
            T.add_edge(a, b)
            in_tree.add(a)
            in_tree.add(b)

    return T


def hub_first_tree(G: nx.Graph, root: Optional[Node] = None) -> nx.Graph:
    """
    find high-degree vertex, maintain front ds of candidate edges crossing from  current tree to unseen vertices,
    always attach the unseen endpoint with highest degree (simple heuristic)
    - We will test if prioritizing hubs will help or hurt cong.
    """
    if root is None:
        root = choose_root_by_degree(G)

    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    in_tree = {root}
    pq: List[Tuple[int, int, Node, Node]] = []
    counter = 0

    for v in G.neighbors(root):
        counter += 1
        heapq.heappush(pq, (-G.degree[v], counter, root, v))

    while len(in_tree) < G.number_of_nodes():
        if not pq:
            raise ValueError("G must be connected.")
        _, _, u, v = heapq.heappop(pq)
        if v in in_tree:
            continue
        T.add_edge(u, v)
        in_tree.add(v)
        for x in G.neighbors(v):
            if x not in in_tree:
                counter += 1
                heapq.heappush(pq, (-G.degree[x], counter, v, x))

    return T


def _bfs_layer_prefix_cut(G: nx.Graph, root: Node, min_balance: float = 0.25) -> Optional[set[Node]]:
    """
    BFS yields layers from root, we try to find a connected side A by taking a prefix such accomodation
    -score candidate prefixes by cut_size / min(|A|, |B|), smaller == better
    -Not a theorem application, just cut-inspired heuristic, seems to be ok
    """
    levels = nx.single_source_shortest_path_length(G, root)
    by_level: Dict[int, List[Node]] = defaultdict(list)
    for u, d in levels.items():
        by_level[d].append(u)

    all_levels = sorted(by_level)
    if len(all_levels) <= 1:
        return None

    n = G.number_of_nodes()
    seen: set[Node] = set()
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


def cut_recursive_tree(G: nx.Graph, root: Optional[Node] = None, min_balance: float = 0.25, fallback_size: int = 10) -> nx.Graph:
    """
    cut-inspired heuristic

    idea:
    1. pick root (default: hub/highest-degree v)
    2. by BFS layers around the root to search for a connected prefix A with relatively small edge boundary and balance
    3. recurse on G[A] and on each connected component of the outside part
    4. connect the resulting trees with one crossing edge per outside component

    inspired by partition-based STC approximations
    """
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")
    if G.number_of_nodes() <= 1:
        T = nx.Graph()
        T.add_nodes_from(G.nodes())
        return T
    if G.number_of_nodes() <= fallback_size or G.number_of_edges() == G.number_of_nodes() - 1:
        return bfs_tree(G, root=root)

    if root is None:
        root = choose_root_by_degree(G)

    A = _bfs_layer_prefix_cut(G, root=root, min_balance=min_balance)
    if A is None or len(A) == 0 or len(A) == G.number_of_nodes():
        return bfs_tree(G, root=root)

    B = set(G.nodes()) - A
    if not B:
        return bfs_tree(G, root=root)

    T = nx.Graph()
    T.add_nodes_from(G.nodes())

    #recursion on A
    TA = cut_recursive_tree(G.subgraph(A).copy(), root=root, min_balance=min_balance, fallback_size=fallback_size)
    T.add_edges_from(TA.edges())

    # recurse on every connected component that cut yielded and connect it back by one edge
    for comp in nx.connected_components(G.subgraph(B)):
        comp = set(comp)
        TC = cut_recursive_tree(G.subgraph(comp).copy(), root=choose_root_by_degree(G.subgraph(comp)), min_balance=min_balance, fallback_size=fallback_size)
        T.add_edges_from(TC.edges())

        crossing_edges = [(u, v) for u in A for v in comp if G.has_edge(u, v)]
        if not crossing_edges:
            raise RuntimeError("No crossing edge found while reconnecting cut-recursive components.")

        # reattaching through a high-degree endpoint in A and a high-degree endpoint in comp if possible
        bridge = max(crossing_edges, key=lambda e: (G.degree[e[0]] + G.degree[e[1]], G.degree[e[0]], G.degree[e[1]]))
        T.add_edge(*bridge)

    validate_tree(G, T)
    return T

def okamoto_decision_simple(G: nx.Graph, k: int, *, return_tree: bool = False) -> tuple[bool, Optional[nx.Graph]]:
    """
    Okamoto-style exact decision DP
    - good[root][mask] = True == (mask, root) is a good rooted subset,
    - where mask is a subset of V \ {root}
    """    
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")

    H, _, back = _relabel_graph_to_ints(G)
    n = H.number_of_nodes()
    _ensure_small_enough(n, MAX_EXACT_N, "okamoto_decision_simple")

    fullmask = (1 << n) - 1
    bit = [1 << v for v in range(n)]
    cut_size, neigh_mask, popcount = _compute_cut_sizes(H)

    masks_by_size: List[List[int]] = [[] for _ in range(n + 1)]
    for mask in range(fullmask + 1):
        masks_by_size[popcount[mask]].append(mask)

    good_prev = [[False] * (fullmask + 1) for _ in range(n)]
    for root in range(n):
        good_prev[root][0] = True

    for size_limit in range(1, n):
        good_curr = [[False] * (fullmask + 1) for _ in range(n)]
        for root in range(n):
            good_curr[root][0] = True

        for size in range(1, size_limit + 1):
            for mask in masks_by_size[size]:
                for root in range(n):
                    if mask & bit[root]:
                        continue
                    #lemma 3.2
                    # Case 1: root attaches to a neighbor u in mask and the rest is good under u.
                    ok = False
                    if cut_size[mask] <= k:
                        nbrs = neigh_mask[root] & mask
                        while nbrs:
                            lsb = nbrs & -nbrs
                            u = lsb.bit_length() - 1
                            smaller = mask ^ lsb
                            if good_prev[u][smaller]:
                                ok = True
                                break
                            nbrs ^= lsb

                    # Case 2: split mask into two non-empty proper parts and join them at root.
                    if not ok:
                        first_bit = mask & -mask
                        sub = (mask - 1) & mask
                        while sub:
                            # symmetry break: keep the smallest bit in sub
                            if (sub & first_bit) and sub != mask:
                                other = mask ^ sub
                                if other and good_prev[root][sub] and good_prev[root][other]:
                                    ok = True
                                    break
                            sub = (sub - 1) & mask

                    good_curr[root][mask] = ok

        good_prev = good_curr

    root = None
    for r in range(n):
        target = fullmask ^ bit[r]
        if good_prev[r][target]:
            root = r
            break

    if root is None:
        return False, None

    if not return_tree:
        return True, None

    def _build_internal_helper(root_v: int, mask: int) -> List[tuple[int, int]]:
        if mask == 0:
            return []

        if cut_size[mask] <= k:
            nbrs = neigh_mask[root_v] & mask
            while nbrs:
                lsb = nbrs & -nbrs
                u = lsb.bit_length() - 1
                smaller = mask ^ lsb
                if good_prev[u][smaller]:
                    edges = _build_internal_helper(u, smaller)
                    edges.append((root_v, u))
                    return edges
                nbrs ^= lsb

        first_bit = mask & -mask
        sub = (mask - 1) & mask
        while sub:
            if (sub & first_bit) and sub != mask:
                other = mask ^ sub
                if other and good_prev[root_v][sub] and good_prev[root_v][other]:
                    return _build_internal_helper(root_v, sub) + _build_internal_helper(root_v, other)
            sub = (sub - 1) & mask

        raise RuntimeError("Could not reconstruct witness tree from DP table.")

    all_edges = _build_internal_helper(root, fullmask ^ bit[root])
    T_int = nx.Graph()
    T_int.add_nodes_from(range(n))
    T_int.add_edges_from(all_edges)

    if T_int.number_of_edges() != n - 1 or not nx.is_tree(T_int):
        raise RuntimeError("Reconstructed object is not a spanning tree.")

    T = nx.relabel_nodes(T_int, back, copy=True)
    validate_tree(G, T)
    return True, T


def okamoto_exact_stc_simple(G: nx.Graph) -> ExactSTCResult:
    start = time.perf_counter()
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")

    H, _, _ = _relabel_graph_to_ints(G)
    n = H.number_of_nodes()
    _ensure_small_enough(n, MAX_EXACT_N, "okamoto_exact_stc_simple")

    cut_size, _, _ = _compute_cut_sizes(H)
    candidates = sorted({value for value in cut_size if value > 0})
    if not candidates:
        T = nx.Graph()
        T.add_nodes_from(G.nodes())
        return ExactSTCResult(0, T, time.perf_counter() - start)

    left = 0
    right = len(candidates) - 1
    best_k = candidates[-1]

    #binary search so optimization problem recieves yes/no
    while left <= right:
        mid = (left + right) // 2
        k = candidates[mid]
        ok, _ = okamoto_decision_simple(G, k, return_tree=False)
        if ok:
            best_k = k
            right = mid - 1
        else:
            left = mid + 1

    ok, T = okamoto_decision_simple(G, best_k, return_tree=True)
    if not ok or T is None:
        raise RuntimeError("Decision said yes, reconstruction failed.")

    return ExactSTCResult(best_k, T, time.perf_counter() - start)


def _exact_balanced_cut_simple(H: nx.Graph) -> Set[Node]:
    """
    compute minimum 2/3-balanced cut by full subset search
    returns one side S of the cut
    exact cut oracle replacing the paper’s polynomial-time approximation oracle,
    """
    if not nx.is_connected(H):
        raise ValueError("H must be connected.")

    G2, _, back = _relabel_graph_to_ints(H)
    n = G2.number_of_nodes()
    _ensure_small_enough(n, MAX_BALANCED_CUT_N, "_exact_balanced_cut_simple")

    if n <= 1:
        return set(H.nodes())

    fullmask = (1 << n) - 1
    cut_size, _, popcount = _compute_cut_sizes(G2)

    lower = math.ceil(n / 3)
    upper = math.floor(2 * n / 3)

    best_mask = None
    best_value = None
    anchor = 1 << 0 #symmetry break

    for mask in range(1, fullmask):
        if not (mask & anchor):
            continue
        size = popcount[mask]
        if size < lower or size > upper:
            continue
        value = cut_size[mask]
        if best_value is None or value < best_value:
            best_value = value
            best_mask = mask

    if best_mask is None:
        best_mask = anchor

    answer = set()
    for v in range(n):
        if best_mask & (1 << v):
            answer.add(back[v])
    return answer


def kolman_congspantree_simple(G: nx.Graph) -> ApproxSTCResult:
    """
    Kolmans-style recursive construction, but with an exact exponential balanced-cut subroutine for clarity purposes
    - if good spanning trees must respect hereditary bisection structure,
    then recursive balanced partitioning should help construct low-congestion trees.
    (STC(G)>=\omega(hb(G)/maxdeg)) where hb is hereditary bisection width

    - partition the graph by a 2/32/3-balanced cut,
    - solve recursively on the resulting connected components,
    - arbitrarily combine the component spanning trees into a spanning tree of the whole graph
    """
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")

    start = time.perf_counter()

    #recursive divide-and-conquer on components
    def _recurse(H: nx.Graph) -> nx.Graph:
        if H.number_of_nodes() == 1:
            T = nx.Graph()
            T.add_nodes_from(H.nodes())
            return T

        side = _exact_balanced_cut_simple(H)
        cut_edges: List[Edge] = []
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
