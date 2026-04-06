import random
from pathlib import Path
from typing import Optional, Iterable
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

#at the same level as src there is data, i need it not to be src/data/thisbunchofgraphs but further inside
# OUTPUT_DIR = Path("data/custom/all_in_one_place_named_approprately").parent
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "custom" / "all_in_one_place_named_appropriately"

def _write_lst(G: nx.Graph, path: Path) -> None:
    mapping = {node: i + 1 for i, node in enumerate(sorted(G.nodes(), key=repr))}
    G = nx.relabel_nodes(G, mapping, copy=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u in sorted(G.nodes()):
            neighbors = " ".join(map(str, sorted(G.neighbors(u))))
            f.write(f"{u}: {neighbors}\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        ("erdos_renyi", lambda i: erdos_renyi_connected(10 + 2*i, 0.30 - 0.02*i, seed=100 + i)),
        ("barabasi_albert", lambda i: barabasi_albert_graph(10 + 2*i, 2 if i < 3 else 3, seed=200 + i)),
        ("star_of_cliques", lambda i: star_of_cliques(3 + (i % 3), 3 if i < 3 else 4)),
        ("two_hub_bridge", lambda i: two_hub_bridge_graph(4 + i, 4 + i, extra_cross_edges=i - 1, seed=300 + i)),
    ]

    graph_id = 1
    for graph_type, make_graph in specs:
        for i in range(1, 6):
            G = make_graph(i)
            out = OUTPUT_DIR / f"graph_{graph_id:03d}_{graph_type}_{i}.lst"
            _write_lst(G, out)
            print(f"wrote {out}")
            graph_id += 1


if __name__ == "__main__":
    main()