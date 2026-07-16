from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple
import networkx as nx

from src.stc_kolman import _compute_cut_sizes
from src.stc_core import validate_tree, _relabel_graph_to_ints, _ensure_small_enough, compute_tree_congestion


Node = Hashable
Edge = Tuple[Node, Node]

MAX_EXACT_N = 19 #3^number so lets stay realistic within bounds of the atoms in universe (more than 20 is pretty bad)

@dataclass
class ExactSTCResult:
    optimum_congestion: int
    tree: nx.Graph
    solve_time_seconds: float

@dataclass
class _OkamotoContext:
    back: Dict[int, Node]
    n: int
    fullmask: int
    bit: List[int]
    cut_size: List[int]
    neigh_mask: List[int]
    popcount: List[int]
    masks_by_size: List[List[int]]

#preprocess al subsets/cut data once
def _build_okamoto_context(G: nx.Graph, limit: int, name: str) -> _OkamotoContext:
    """
    Prepares the exact DP universe:
    - relabels vertices to integers, & checks if n is below the exact threshold,
    - precomputes all subset cut sizes,
    - precomputes subset sizes,
    - groups masks by cardinality, dp will be doing bottom up by subset cardinality
    """

    #checkers
    if not nx.is_connected(G): raise ValueError("G must be connected.")
    H, _, back = _relabel_graph_to_ints(G)
    n = H.number_of_nodes()
    _ensure_small_enough(n, limit, name)

    cut_size, neigh_mask, popcount = _compute_cut_sizes(H)

    masks_by_size = [[] for _ in range(n + 1)]
    for S in range(2**n):
        masks_by_size[popcount[S]].append(S)

    return _OkamotoContext(
        back=back,
        n=n,
        fullmask=2**n-1,
        bit=[2**v for v in range(n)],
        cut_size=cut_size,
        neigh_mask=neigh_mask,
        popcount=popcount,
        masks_by_size=masks_by_size,
    )

#solve the decision problem stc(G) ≤ k using DP over states good[root][S]
def _okamoto_decision_from_context(ctx: _OkamotoContext,G: nx.Graph, k: int, *, return_tree: bool = True) -> tuple[bool, Optional[nx.Graph]]:
    """
    solves question "Is there a spanning tree of G with maximum congestion <= k"
    - good[root][S] = True == (S, root) is a good rooted subset,
    - where S is a subset of V \\ {root}

    can I build a valid rooted tree on {root} ∪ S under congestion bound k?
    A state is true in only two ways:
    1. attach one child subtree; pick a neighbor u of root inside S, and if good[u][S \ {u}] is true, then connect root to u
    2. split at the same root; split S into two smaller parts A and other; if both good[root][A] and good[root][other] are true, then combine them under the same root
    """
    n, fullmask = ctx.n, ctx.fullmask

    #base case, S=0 root is always feasable
    good = [bytearray(fullmask + 1) for _ in range(n)] #says state exists
    for root in range(n):
        good[root][0] = True

    choice_kind: Optional[List[bytearray]] = None
    choice_arg: Optional[List[List[int]]] = None
    if return_tree:
        choice_kind = [bytearray(fullmask + 1) for _ in range(n)] #either case 1 or 2 was used to obtain this dp state, by state 1/2 transition
        choice_arg = [[-1] * (fullmask + 1) for _ in range(n)] #“which child” or “which split” parameter was used by transition

    #dp step
    for size in range(1, n):
        for S in ctx.masks_by_size[size]: #larger states depend on smaller, bottom up DP bysubset size

            for root in range(n): # for each subset S dp tries every possible root that is not inside subset
                #as it is defined S being outside root...
                if S & ctx.bit[root]:
                    continue

                #case1: look at neghbor u of root that lies inside S, try to make u the root of the rest of the subtree
                #if good[u][S \ {u}] is true, then you can connect root to u and obtain a feasible rooted tree for (root, S)
                #(u,S\{u}) implies (root,S)
                found = False
                nbrs = ctx.neigh_mask[root] & S #only eligible nodes to be attached to root
                while nbrs:
                    lsb = nbrs & -nbrs
                    u = lsb.bit_length() - 1 #tak vertex u that lies inside S and is adjecent to root
                    smaller = S ^ lsb #smaller=S\{u}
                    if ctx.cut_size[S] <= k  and good[u][smaller]:
                        good[root][S] = 1 # we can connect u to root and get valid rooted tree for (root,S)
                        if return_tree and choice_kind is not None and choice_arg is not None:
                            choice_kind[root][S] = 1
                            choice_arg[root][S] = u
                        found = True
                        break
                    nbrs ^= lsb #removes actual and goes to next

                if found: continue
                #if case 2 failed, dp asks: can the rooted tree for (root, S) be obtained by
                #combining 2 smaller rooted trees that share the same root? HENCE
                #case 2: split S into two non-empty proper parts (A and other) and join them at root.
                A = (S - 1) & S
                first_bit = S & -S #lsb
                while A: #enum all subsets of masks
                    if A & first_bit: #symmetry break...
                        other = S ^ A #xor gives other=S\A
                        if other and good[root][A] and good[root][other]:
                            good[root][S] = 1
                            if return_tree and choice_kind is not None and choice_arg is not None:
                                choice_kind[root][S] = 2
                                choice_arg[root][S] = A
                            break
                    A = (A - 1) & S

    #after filling the table we check if there is some root r such that
    root = None
    for r in range(n):
        target = fullmask ^ ctx.bit[r]
        if good[r][target]:
            root = r
            break

    if root is None: return False, None
    if not return_tree: return True, None


    #dp proved existence, this builds witness == spanning tree
    assert choice_kind is not None and choice_arg is not None
    def _build_internal_helper(root_v: int, S: int) -> List[tuple[int, int]]:
        """
        The recursive function is meant to return
        all edges of a rooted tree on the vertex set {root_v} ∪ vertices(S)
        where rooted tree is exactly the one whose existence was certified by good[root_v][S] = True
        """
        if S == 0: return [] #just root v, has no edges

        kind = choice_kind[root_v][S] #how the state was made true
        arg = choice_arg[root_v][S]

        #reconstruct child subtree and appends edge (root,u),
        if kind == 1: #attach one child rule, hence arg stores that child u
            u = arg #choose a neighbor u of root_v inside S
            smaller = S ^ ctx.bit[u] #(u was inside S, in recurrence its peeled out of S and promoted to be the child root of recursive subtree)
            edges = _build_internal_helper(u, smaller) #prove good[u][S \ {u}], recursion to get all child subtree
            edges.append((root_v, u)) #conlcude good[root_v][S]
            return edges

        #reconstruct the two root-sharing subpieces and unions their edges
        if kind == 2: #split at root rule, hence arg has one side of the split
            A = arg
            other = S ^ A
            return _build_internal_helper(root_v, A) + _build_internal_helper(root_v, other) #concat edgelist of 2 recuriseive  {root_v} union other and {root_v} union A

        raise RuntimeError("Could not reconstruct witness tree from stored DP choices.")

    #checks if Tree
    all_edges = _build_internal_helper(root, fullmask ^ ctx.bit[root])
    T_int = nx.Graph()
    T_int.add_nodes_from(range(n))
    T_int.add_edges_from(all_edges)

    if T_int.number_of_edges() != n - 1 or not nx.is_tree(T_int):
        raise RuntimeError("Reconstructed object is not spanning tree...")

    T = nx.relabel_nodes(T_int, ctx.back, copy=True)
    validate_tree(G, T)
    return True, T

#decision solver is turned into exact optimizer by binary search over feasable cut vals
def okamoto_exact_stc_simple(G: nx.Graph) -> ExactSTCResult:
    """turns decision DP into optimization solver by binary searching on candidate congestion values

    1. build context (preprocessing)
    2. collect all positive values appearing in ctx.cut_size, sort them
    3. binary-search the minimum k for which the decision DP says “yes”,
    4. rerun once with return_tree=True to reconstruct a witness tree

    The optimum congestion must equal the congestion of some edge in an optimal tree,
    and every tree-edge congestion is a cut size of some vertex partition.
    So searching over realizable subset boundary sizes is the discrete candidate set the code uses.
    That matches the STC cut-based definition.
    """

    start = time.perf_counter()

    #edgecase - tree
    if G.number_of_edges() == G.number_of_nodes() - 1 and nx.is_connected(G):
        T = G.copy()
        return ExactSTCResult(1, T, time.perf_counter() - start)

    #edgecase - 1cycle with leaves
    if G.number_of_edges() == G.number_of_nodes() and nx.is_connected(G):
            cycles = nx.cycle_basis(G)
            if len(cycles) == 1:
                cycle_nodes = cycles[0]
                cycle_edges = list(zip(cycle_nodes, cycle_nodes[1:] + cycle_nodes[:1]))

                best_congestion = None
                best_T = None
                for u, v in cycle_edges:
                    T_candidate = G.copy()
                    T_candidate.remove_edge(u, v)
                    congestion = compute_tree_congestion(G, T_candidate)
                    if best_congestion is None or congestion.max_congestion < best_congestion:
                        best_congestion = congestion.max_congestion
                        best_T = T_candidate

                if best_T is not None:
                    return ExactSTCResult(best_congestion, best_T, time.perf_counter() - start)

    ctx = _build_okamoto_context(G, MAX_EXACT_N, "okamoto_exact_stc_simple")
    #because congestion of a tree edge is a cut size,
    #optimum congestion must be one of the realizable subset boundary values, thts why candidate set is:
    candidates = sorted({value for value in ctx.cut_size if value > 0})
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
        ok, _ = _okamoto_decision_from_context(ctx, G, k, return_tree=False)
        if ok:
            best_k = k
            right = mid - 1
        else:
            left = mid + 1

    ok, T = _okamoto_decision_from_context(ctx, G, best_k, return_tree=True)
    if not ok or T is None:
        raise RuntimeError("Decision said yes, reconstruction failed.")

    return ExactSTCResult(best_k, T, time.perf_counter() - start)

