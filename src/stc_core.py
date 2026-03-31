from __future__ import annotations

from collections import deque
from dataclasses import  dataclass
from typing import Callable, Dict, Hashable, Optional, Tuple, Iterable, List
import networkx as nx

Node = Hashable
Edge = Tuple[Node, Node]
TreeBuilder = Callable[[nx.Graph], nx.Graph]

@dataclass
class CongestionResult:
    edge_congestion: Dict[Edge, int]
    max_congestion: int


#helper
def canon_edge(u: Node, v: Node) -> tuple[Hashable]:
    """Canonical undir edge representation, so we are consistent"""
    return tuple(sorted((u, v)))

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
    parent= {root: None}
    depth = {root: 0}
    order = []
    q = deque([root])
    while q: #quick bfs to layer out the graph so lca queries are faster
        u = q.popleft()
        order.append(u)
        for v in T.neighbors(u):
            #conditions to stop walking backwards or reprocessing nodes
            if v == parent[u]:continue
            if v in parent: continue

            parent[v] = u
            depth[v] = depth[u] + 1
            q.append(v)

    if len(order) != T.number_of_nodes(): #quick G connectivity check
        raise ValueError("T is not connected.")

    lg = max(1, len(order).bit_length())
    #jumping upwards in node hierarchy with size step in powers of 2
    up = {u: [None] * lg for u in T.nodes()} #up[u][0] = parent, up[u][1] = grandparent, up[u][2] = 4-th ancestor...
    for u in T.nodes():
        up[u][0] = parent[u]
    for j in range(1, lg):
        for u in T.nodes():
            mid = up[u][j - 1]
            up[u][j] = None if mid is None else up[mid][j - 1]

    return parent, depth, order, up, lg

#helper
def _lca(u: Node, v: Node, depth, up, lg: int) -> Node:
    if depth[u] < depth[v]: #just make u deeper one for better consistent reference
        u, v = v, u

    diff = depth[u] - depth[v] #diff = 13 = 8 + 4 + 1, jump by ancestors 2^0, 2^2, 2^3
    bit = 0
    while diff:
        if diff & 1:
            u = up[u][bit]
        diff >>= 1
        bit += 1

    if u == v: return u #if already ancestor relationship

    for j in range(lg - 1, -1, -1): #finds highest level where ancestors still differ and jumps both up so they are children of their lca
        if up[u][j] != up[v][j]:
            u = up[u][j]
            v = up[v][j]

    return up[u][0]


def compute_tree_congestion(G: nx.Graph, T: nx.Graph, root: Optional[Node] = None) -> CongestionResult:
    """
    computes congestion of every tree edge in O(m log n)
    O(n) extra per bottom-up accumulation
    lca query per edge is O(logn)

    For each graph edge {u,v} in G, add +1 to every tree edge along the unique u-v path in T,
    (doing that for each path is slow), hence we
    add +1 at u
    add +1 at v
    subtract 2 at lca(u,v)

    then bottom-up pass accumulates child values onto parents.

    if we sum per-node deltas over all nodes in S, we get 1 for every graph edge that crosses S and 0 otherwise.
    case1: uv inside S & lca inside S -> +1+1-2=0
    case2: uv outside S -> trivially 0
    case3: u inside S, v outisde S -> w=lca(u,v) must be outside S so -2 to outside S, hence sum over S = +1
    hence we count this edge as crossing in S

    For every non-root node x, accumulated val is eq to number of graph-edge paths
    crossing in the tree edge (x, parent[x]) === congestion of that tree edge
    """
    validate_tree(G, T)
    if root is None:
        root = next(iter(T.nodes()))

    parent, depth, order, up, lg = _prepare_rooted_tree(T, root)

    #counting invariant
    delta = {u:0 for u in T.nodes()}
    for u, v in G.edges():
        w = _lca(u, v, depth, up, lg)
        delta[u] += 1
        delta[v] += 1
        delta[w] -= 2

    #bottom up accumulation O(n)
    edge_congestion = {}
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

#primitive stats for csv
def graph_statistics(G: nx.Graph) -> Dict[str, float]:
    return {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "max_degree": max((deg for _, deg in G.degree()), default=0),
        "density": nx.density(G),
    }

######kolmans+okamotos helpers


#helper
def _compute_cut_sizes(H: nx.Graph) -> tuple[List[int], List[int], List[int]]:
    """
    For every subset mask ⊆ V(H), it computes:
    - cut_size[mask] = number of graph edges crossing from mask to its complement
    - neigh_mask[v] = bitmask of neighbors of vertex v
    - popcount[mask] = size of subset

    cut(mask) = cut(rest) + deg(v) − 2|N(v)&rest|
    comes from adding one vertex v into already known subset rest
    - inittially v contributes deg(v) crossing edges
    - but every neighbor already inside rest was counted as crossing, hence must be removed 2times from boundary count,
        once per cut_size[rest] and once from degree[v]

    without precomputing each cut_size(mask), _exact_balanced_cut_simple would require recomputing the boundary from scratch -> O(m) extra
    """
    n = H.number_of_nodes()
    degree, neigh_mask = [0] * n, [0] * n
    for v in range(n):
        degree[v] = H.degree[v]
    for u, v in H.edges():
        neigh_mask[u] |= 2**v #append in the list basically...
        neigh_mask[v] |= 2**u

    cut_size = [0] * (2**n)
    for mask in range(1, 2**n):
        lsb = mask & -mask #least significant bit access
        v = lsb.bit_length() - 1 #converted to vertex index
        rest = mask ^ lsb #rmoves vertex from the subset
        inside_neighbors = (neigh_mask[v] & rest).bit_count() #counts how many neighbors of v already lie inside rest, ∣N(v) & rest∣
        cut_size[mask] = cut_size[rest] + degree[v] - 2 * inside_neighbors

    return cut_size, neigh_mask, _all_popcounts(2**n -1)

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
def _all_popcounts(fullmask: int) -> List[int]:
    pop = [0] * (fullmask + 1)
    for mask in range(1, fullmask + 1):
        pop[mask] = pop[mask >> 1] + (mask & 1)
    return pop


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


#####3 bfs baseline
#tree builder
def bfs_tree(G: nx.Graph, root: Optional[Node] = None) -> nx.Graph:
    """Baseline: cheap runtime + guaranteed spanning tree (but not minimal), sanity check..."""
    if root is None:
        root = choose_root_by_degree(G)
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    T.add_edges_from(nx.bfs_edges(G, root))
    return T
