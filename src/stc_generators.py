import random
from typing import Optional
import networkx as nx


def erdos_renyi_connected(n: int, p: float, seed: Optional[int] = None, max_tries: int = 100) -> nx.Graph:
    rng = random.Random(seed)
    for _ in range(max_tries):
        G = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9))
        if nx.is_connected(G):
            return G
    raise RuntimeError("Failed to generate a connected G(n,p) graph.")


def barabasi_albert_graph(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)


def star_of_cliques(num_cliques: int, clique_size: int) -> nx.Graph:
    G = nx.Graph()
    hub = "hub"
    G.add_node(hub)

    for i in range(num_cliques):
        clique_nodes = [f"c{i}_{j}" for j in range(clique_size)]
        G.add_nodes_from(clique_nodes)
        for u in clique_nodes:
            G.add_edge(hub, u)
        for a in range(clique_size):
            for b in range(a + 1, clique_size):
                G.add_edge(clique_nodes[a], clique_nodes[b])
    return G


def two_hub_bridge_graph(left_size: int, right_size: int, extra_cross_edges: int = 0, seed: Optional[int] = None) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.Graph()
    h1, h2 = "h1", "h2"
    G.add_nodes_from([h1, h2])

    left = [f"L{i}" for i in range(left_size)]
    right = [f"R{i}" for i in range(right_size)]
    G.add_nodes_from(left + right)

    for u in left:
        G.add_edge(h1, u)
    for u in right:
        G.add_edge(h2, u)

    for i in range(left_size):
        for j in range(i + 1, left_size):
            G.add_edge(left[i], left[j])
    for i in range(right_size):
        for j in range(i + 1, right_size):
            G.add_edge(right[i], right[j])

    G.add_edge(h1, h2)

    all_pairs = [(u, v) for u in left for v in right]
    rng.shuffle(all_pairs)
    for u, v in all_pairs[:extra_cross_edges]:
        G.add_edge(u, v)

    return G
