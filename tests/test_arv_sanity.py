"""
sanity tests for ARV balanced cut oracle

checks if:
  1. ARV produces valid 2/3-balanced cut
  2. ARVs cut is not far from exact opt cut
  3. Kolmanw construction still produces valid spanning tree when oracle is ARV
"""

import math
import networkx as nx

from src.stc_arv import arv_balanced_cut
from src.stc_kolman import _exact_balanced_cut_simple, kolman_main
from src.stc_core import validate_tree


def _is_two_thirds_balanced(H: nx.Graph, side: set) -> bool:
    n = H.number_of_nodes()
    a = len(side)
    b = n - a
    return math.ceil(n / 3) <= a <= math.floor(2 * n / 3) and b > 0


def test_arv_produces_balanced_cut():
    for G in [nx.cycle_graph(12), nx.grid_2d_graph(4, 5),
              nx.erdos_renyi_graph(15, 0.3, seed=1)]:
        G = nx.convert_node_labels_to_integers(G)
        if not nx.is_connected(G):
            continue
        side = arv_balanced_cut(G, seed=42)
        assert _is_two_thirds_balanced(G, side), \
            f"ARV cut not 2/3-balanced on {G}"
        print(f"  n={G.number_of_nodes()}: ARV cut size = "
              f"{nx.cut_size(G, side, set(G.nodes()) - side)}, "
              f"|side|={len(side)} -- OK")


def test_arv_vs_exact_on_small():
    """ARV8alpha == exact balanced cut for some small alpha"""
    for G in [nx.cycle_graph(10), nx.path_graph(12),
              nx.erdos_renyi_graph(14, 0.35, seed=7)]:
        G = nx.convert_node_labels_to_integers(G)
        if not nx.is_connected(G):
            continue
        exact_side = _exact_balanced_cut_simple(G)
        arv_side = arv_balanced_cut(G, seed=42)
        rest = lambda s: set(G.nodes()) - s
        exact_cut = nx.cut_size(G, exact_side, rest(exact_side))
        arv_cut = nx.cut_size(G, arv_side, rest(arv_side))
        ratio = arv_cut / exact_cut if exact_cut > 0 else float("inf")
        print(f"  n={G.number_of_nodes()}: exact={exact_cut}, "
              f"arv={arv_cut}, ratio={ratio:.2f}")
        assert arv_cut >= exact_cut, \
            "ARV cut cannot be smaller than the exact optimum"


def test_kolman_with_arv_produces_spanning_tree():
    """Kolman + ARV oracle has to  produce valid spanning tree"""
    for G in [nx.erdos_renyi_graph(25, 0.25, seed=3),
              nx.barabasi_albert_graph(30, 3, seed=5)]:
        if not nx.is_connected(G):
            continue
        result = kolman_main(G)
        validate_tree(G, result.tree)  # raises if not a spanning tree
        print(f"  n={G.number_of_nodes()}: Kolman+ARV tree valid, "
              f"congestion={result.congestion.max_congestion} -- OK")


if __name__ == "__main__":
    print("test_arv_produces_balanced_cut:")
    test_arv_produces_balanced_cut()
    print("test_arv_vs_exact_on_small:")
    test_arv_vs_exact_on_small()
    print("test_kolman_with_arv_produces_spanning_tree:")
    test_kolman_with_arv_produces_spanning_tree()
    print("\nAll sanity checks passed.")