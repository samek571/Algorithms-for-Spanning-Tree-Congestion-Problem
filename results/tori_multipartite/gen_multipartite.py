from pathlib import Path
import networkx as nx
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "multipartite"
MAX_N = 100

def _write_lst(G: nx.Graph, path: Path) -> None:
    mapping = {node: i + 1 for i, node in enumerate(sorted(G.nodes(), key=repr))}
    G = nx.relabel_nodes(G, mapping, copy=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u in sorted(G.nodes()):
            neighbors = " ".join(map(str, sorted(G.neighbors(u))))
            f.write(f"{u}: {neighbors}\n")


def part_lists():
    """sorted part size lists, no isomorphic duplicates."""
    seen = set()
    candidates = []

    # balanced K_{t,t}, K_{t,t,t}, K_{t,t,t,t}
    for k in (2, 3, 4):
        for t in range(2, MAX_N):
            parts = [t] * k
            if sum(parts) <= MAX_N:
                candidates.append(parts)

    # some unbalanced - bipartite skewed and mixed tripartite
    for a, b in [(2, 10), (2, 20), (3, 15), (5, 25), (2, 40), (10, 30)]:
        candidates.append([a, b])
    for a, b, c in [(2, 3, 5), (2, 5, 10), (3, 5, 7), (5, 10, 15), (2, 2, 20)]:
        candidates.append([a, b, c])

    for parts in candidates:
        key = tuple(sorted(parts))
        if key in seen:
            continue
        if sum(parts) <= MAX_N:
            seen.add(key)
            yield list(key)


def gen():
    OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for parts in part_lists():
        G = nx.complete_multipartite_graph(*parts)
        label = "K_" + "_".join(map(str, parts))
        _write_lst(G, OUT / f"{label}.lst")
        count += 1
        print(f"{label:16s} n={G.number_of_nodes():3d} m={G.number_of_edges()}")
    print(f"\nwrote {count} graphs -> {OUT}")


if __name__ == "__main__":
    gen()
