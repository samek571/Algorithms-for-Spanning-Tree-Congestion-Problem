from pathlib import Path
import networkx as nx
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "tori"
MAX_N = 100  # total vertices m*n
MIN_DIM = 3 # base case m, n >= 3


def _write_lst(G: nx.Graph, path: Path) -> None:
    mapping = {node: i + 1 for i, node in enumerate(sorted(G.nodes(), key=repr))}
    G = nx.relabel_nodes(G, mapping, copy=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u in sorted(G.nodes()):
            neighbors = " ".join(map(str, sorted(G.neighbors(u))))
            f.write(f"{u}: {neighbors}\n")



def dims():
    """ m <= n torus dimensions with m*n <= MAX_N."""
    seen = set()
    for m in range(MIN_DIM, MAX_N):
        for n in range(m, MAX_N): #n >= n so there should not be iso
            if m * n <= MAX_N:
                key = (m, n)
                if key not in seen:
                    seen.add(key)
                    yield m, n


def gen():
    OUT.mkdir(parents=True, exist_ok=True)
    res = 0
    for m, n in dims():
        G = nx.cartesian_product(nx.cycle_graph(m), nx.cycle_graph(n))
        _write_lst(G, OUT / f"torus_{m}x{n}.lst")
        res += 1
        print(f"torus_{m}x{n}  nodes={G.number_of_nodes():3d} "
              f"stc={2*min(m,n)}")
    print(f"\nwrote {res} graphs -> {OUT}")


if __name__ == "__main__":
    gen()
