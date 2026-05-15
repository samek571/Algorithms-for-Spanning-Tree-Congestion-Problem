"""
ARV balanced-cut oracle used inside Kolmans recursive STC heuristic paper

Arora-Rao-Vazirani algorithm for 2/3-balanced cut:
- solve ARV SDP relaxation (unit vectors + spreading + triangle inequalities)
- round by projection into random directions and sweeping prefixes
- return cheapest 2/3-balanced cut fount across all projections

Provides the O(sqrt(log n)) approximation guarantee required by Theorem 1 of
Kolman 2025, when used as the cut oracle inside the recursive construction.
"""
from __future__ import annotations

import math
from typing import Optional, Set, Hashable, List, Tuple

import numpy as np
import networkx as nx

try:
    import cvxpy as cp
except ImportError as e:
    raise ImportError(
        "stc_arv requires cvxpy. Install with: pip install cvxpy"
    ) from e

Node = Hashable


def _solve_arv_sdp(H: nx.Graph) -> np.ndarray:
    """
    solving ARV SDP relaxation for H

    Decision variable: gram matrix X : X[i][j] = <v_i, v_j>.
    Constraints:
      - X is PSD hence  X[i][j] = <v_i, v_j>
      - X[i][i] = 1 vector
      - X[i][i] + X[j][j] - 2*X[i][j] = ||v_i - v_j||^2
        and these distances fulfill  triangle ineq:
        ||v_i - v_j||^2 + ||v_j - v_k||^2 >= ||v_i - v_k||^2 for all i,j,k
      - spreading: sum over all i<j of ||v_i - v_j||^2 >= n^2 / 4
    Objective: min sum over edges (i,j) of ||v_i - v_j||^2

    Returns vector embedding (n, n) sd numpy array, where row i is v_i
    recovered by Choleski-like factorization of X
    """
    nodes = list(H.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}

    #gram matrix vars
    X = cp.Variable((n, n), symmetric=True)

    constraints = [X >> 0]  # PSD constraint
    constraints += [X[i, i] == 1 for i in range(n)]  # unit vectors

    # Squared distance: d[i,j] = X[i,i] + X[j,j] - 2 X[i,j] = 2 - 2 X[i,j]
    # Triangle inequality: d[i,j] + d[j,k] >= d[i,k] for all triples
    # same as: (2 - 2 X[i,j]) + (2 - 2 X[j,k]) >= (2 - 2 X[i,k]) equiv  X[i,j] + X[j,k] - X[i,k] <= 1.
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                constraints.append(X[i, j] + X[j, k] - X[i, k] <= 1)

    # Spreading constraint: sum of squared distances >= n^2 / 4.
    # sum_{i<j} (2 - 2 X[i,j]) >= n^2 / 4
    # equiv:  sum_{i<j} X[i,j] <= n*(n-1)/2 - n^2/8
    sum_offdiag = cp.sum(X) - cp.trace(X)  # 2 * sum_{i<j} X[i,j]
    constraints.append(sum_offdiag <= n * (n - 1) - n * n / 4)

    # Objective: minimize sum over edges of squared distances
    edge_distance_terms = []
    for u, v in H.edges():
        i, j = idx[u], idx[v]
        edge_distance_terms.append(2 - 2 * X[i, j])
    if edge_distance_terms:
        objective = cp.Minimize(cp.sum(edge_distance_terms))
    else:
        objective = cp.Minimize(0)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if X.value is None:
        raise RuntimeError(f"ARV SDP failed to solve. Status: {problem.status}")

    # recover vectors from Gram matrix via decompositions of eigens
    # X = V V^T where rows of V are the v_i
    X_value = np.asarray(X.value)
    # symm numerically
    X_value = (X_value + X_value.T) / 2.0
    # negative eigenvalues => 0 as they dont help
    eigenvalues, eigenvectors = np.linalg.eigh(X_value)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    vectors = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]
    return vectors  # shape (n, n); row i is v_i


def _sweep_balanced_prefix_cut(
        H: nx.Graph,
        ordering: List[Node],
) -> Optional[Tuple[Set[Node], int]]:
    """
    takes ordering of vertices, evaluates all prefix cuts
    returns cheapest one satisfying the 2/3-balance condition that is (side, cut_size) or None
    """
    n = H.number_of_nodes()
    if n <= 1: return None

    lower = math.ceil(n / 3)
    upper = math.floor(2 * n / 3)
    if lower > upper: return None

    all_nodes = set(ordering)
    side: Set[Node] = set()
    best: Optional[Tuple[Set[Node], int]] = None

    for i, u in enumerate(ordering[:-1], start=1):
        side.add(u)
        if i < lower or i > upper: continue

        size = nx.cut_size(H, side, all_nodes - side)
        if best is None or size < best[1]:
            best = (set(side), size)

    return best


def arv_balanced_cut(
        H: nx.Graph,
        num_projections: Optional[int] = None,
        seed: Optional[int] = None,
) -> Set[Node]:
    """
    ARV balanced-cut oracle;
    - solves ARV SDP,
    - performs logn random projection rounds
    - returns one side of the cheapest 2/3-balanced sweep cut found in the run

    By THM1 of Kolman 2025, the returned cut has cost at most O(sqrt(log n)) * OPT bisection cost (its randomized algo, so we can only state such being true with high probab)
    """
    if not nx.is_connected(H): raise ValueError("H must be connected.")

    n = H.number_of_nodes()
    if n <= 1: return set(H.nodes())

    if num_projections is None:
        num_projections = max(5, int(math.ceil(math.log2(max(n, 2)))))

    rng = np.random.default_rng(seed)

    nodes = list(H.nodes())
    vectors = _solve_arv_sdp(H)

    best_side: Optional[Set[Node]] = None
    best_size: Optional[int] = None

    for _ in range(num_projections):
        #rng  unit direction in R^n
        g = rng.standard_normal(n)
        g /= np.linalg.norm(g)
        #project each v
        projections = vectors @ g  # shape (n,)
        # order vertices by projection value
        order_fwd = [nodes[i] for i in np.argsort(projections)]
        order_rev = list(reversed(order_fwd))

        for ordering in (order_fwd, order_rev):
            result = _sweep_balanced_prefix_cut(H, ordering)
            if result is None:
                continue
            side, size = result
            if best_size is None or size < best_size:
                best_side = side
                best_size = size

    if best_side is None:
        raise RuntimeError(
            f"ARV oracle found no balanced cut on graph with n={n}. "
            "bug or malformed input present...")

    return best_side

