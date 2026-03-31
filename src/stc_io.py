from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx


@dataclass
class GraphInstance:
    graph: nx.Graph
    graph_label: str
    graph_family: str
    source_path: str

#removes blank lines and full in line comments
def _is_comment_or_empty(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in {"#", "%", "//"})

def _strip_inline_comment(line: str) -> str:
    body = line
    for prefix in {"#", "%", "//"}:
        pos = body.find(prefix)
        if pos != -1:
            body = body[:pos]
    return body.strip()


#house of graph adjecency list parser
def _parse_adjlist_line(line: str) -> tuple[int, list[int]]:
    body = _strip_inline_comment(line)
    if not body:
        raise ValueError("empty file...")
    if ":" not in body:
        raise ValueError(f"Expected format 'u: v1 v2 ...', received: {line!r}")

    left, right = body.split(":", 1)
    u = int(left.strip())
    neighbors = [int(tok) for tok in right.split()]
    return u, neighbors


def load_adjlist_graph(path: str | Path, *, remove_self_loops: bool = True) -> nx.Graph:
    path = Path(path)
    G = nx.Graph()

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            if _is_comment_or_empty(raw_line):
                continue
            u, neighbors = _parse_adjlist_line(raw_line)
            G.add_node(u)
            for v in neighbors:
                if remove_self_loops and u == v:
                    continue
                G.add_edge(u, v)

    return G


def normalize_graph_labels(G: nx.Graph) -> nx.Graph:
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)


def _graph_family_from_path(path: Path, root: Path | None) -> str:
    if root is None:
        return path.parent.name if path.parent.name else "."
    try:
        rel_parent = path.parent.relative_to(root)
    except ValueError:
        return path.parent.name if path.parent.name else "."
    rel_str = rel_parent.as_posix()
    return rel_str if rel_str else "."


def load_graph_file(
    path: str | Path,
    *,
    root: str | Path | None = None,
    normalize_labels_flag: bool = False,
) -> GraphInstance:
    path = Path(path)
    root_path = None if root is None else Path(root)
    G = load_adjlist_graph(path)

    if normalize_labels_flag:
        G = normalize_graph_labels(G)

    return GraphInstance(
        graph=G,
        graph_label=path.stem,
        graph_family=_graph_family_from_path(path, root_path),
        source_path=str(path.resolve()),
    )


def iter_graph_files(root: str | Path, *, suffixes = {".lst", ".txt"}) -> Iterable[Path]:
    root = Path(root)
    candidates = root.rglob("*")
    lowered = tuple(s.lower() for s in suffixes)

    files: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in lowered:
            continue
        files.append(path)

    files.sort(key=lambda p: (p.parent.as_posix(), p.name))
    return files


def load_graph_folder(root: str | Path, *, suffixes = {".lst", ".txt"}, normalize_labels_flag: bool = False) -> list[GraphInstance]:
    root = Path(root)
    instances: list[GraphInstance] = []
    for path in iter_graph_files(root, suffixes=suffixes):
        instance = load_graph_file(
            path,
            root=root,
            normalize_labels_flag=normalize_labels_flag,
        )
        instances.append(instance)
    return instances


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "data"
    if root.exists():
        instances = load_graph_folder(root)
        print(f"Project root: {project_root}")
        print(f"Data root: {root}")
        print(f"Found {len(instances)} candidate graph files under {root}")
    else:
        print(f"Path does not exist: {root}")
