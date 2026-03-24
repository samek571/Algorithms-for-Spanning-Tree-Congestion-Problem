from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import networkx as nx


@dataclass
class GraphInstance:
    graph: nx.Graph
    graph_label: str
    graph_family: str
    source_path: str


_DEFAULT_SUFFIXES = (".adj", ".adjlist", ".alist", ".lst", ".txt")
_COMMENT_PREFIXES = ("#", "%", "//")


def _is_comment_or_empty(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in _COMMENT_PREFIXES)


def _strip_inline_comment(line: str) -> str:
    out = line
    for prefix in _COMMENT_PREFIXES:
        pos = out.find(prefix)
        if pos != -1:
            out = out[:pos]
    return out.strip()


def _coerce_token(token: str):
    try:
        return int(token)
    except ValueError:
        return token


def _normalize_token_text(text: str) -> str:
    text = text.replace("->", " ")
    text = text.replace(":", " ")
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    text = text.replace("|", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    return text


def _tokenize_line(line: str) -> list[str]:
    text = _normalize_token_text(_strip_inline_comment(line))
    return [tok for tok in re.split(r"\s+", text.strip()) if tok]


def _parse_labeled_adjlist_line(line: str):
    tokens = _tokenize_line(line)
    if not tokens:
        return None
    vertex = _coerce_token(tokens[0])
    neighbors = [_coerce_token(tok) for tok in tokens[1:]]
    return vertex, neighbors


def _looks_like_vertex_count_line(line: str) -> int | None:
    tokens = _tokenize_line(line)
    if len(tokens) != 1:
        return None
    tok = tokens[0]
    if re.fullmatch(r"[0-9]+", tok):
        return int(tok)
    return None


def _contains_explicit_vertex_marker(line: str) -> bool:
    stripped = _strip_inline_comment(line)
    return (":" in stripped) or ("->" in stripped)


def load_adjlist_graph(path: str | Path, *, remove_self_loops: bool = True) -> nx.Graph:
    """Load an undirected simple graph from an adjacency-list file.

    Supported styles:
    1. Labeled lines:
         1: 2 3 4
         a -> b c d
         7 9 10 11   # interpreted as vertex 7 with neighbors 9,10,11

    2. Header + unlabeled adjacency lines (common in some .lst exports):
         n
         2 5
         1 3
         ...
       which is interpreted as vertices 1..n, one adjacency list per line.
    """
    path = Path(path)
    G = nx.Graph()

    with path.open("r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    lines = [line for line in raw_lines if not _is_comment_or_empty(line)]
    if not lines:
        return G

    header_n = _looks_like_vertex_count_line(lines[0])
    use_header_mode = False
    if header_n is not None and len(lines) >= header_n + 1:
        # Only switch into header mode if the next header_n lines do not appear to
        # be explicitly labeled by ':' or '->'. This keeps ordinary labeled files safe.
        next_lines = lines[1 : 1 + header_n]
        if next_lines and not any(_contains_explicit_vertex_marker(line) for line in next_lines):
            use_header_mode = True

    if use_header_mode:
        n = header_n
        G.add_nodes_from(range(1, n + 1))
        body = lines[1 : 1 + n]
        for idx, line in enumerate(body, start=1):
            tokens = _tokenize_line(line)
            if not tokens:
                continue
            # If the line clearly starts with its own label, drop that first token.
            if tokens and re.fullmatch(r"[0-9]+", tokens[0]):
                first = int(tokens[0])
                if first == idx:
                    tokens = tokens[1:]
            for tok in tokens:
                v = _coerce_token(tok)
                if remove_self_loops and idx == v:
                    continue
                G.add_edge(idx, v)
        return G

    # Default: one labeled adjacency entry per line.
    for line in lines:
        parsed = _parse_labeled_adjlist_line(line)
        if parsed is None:
            continue
        u, neighbors = parsed
        G.add_node(u)
        for v in neighbors:
            if remove_self_loops and u == v:
                continue
            G.add_edge(u, v)

    return G


def normalize_graph_labels(G: nx.Graph) -> nx.Graph:
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)


def graph_name_from_path(path: str | Path) -> str:
    return Path(path).stem


def graph_family_from_path(path: str | Path, root: str | Path | None = None) -> str:
    path = Path(path)
    if root is None:
        return path.parent.name if path.parent.name else "."

    root = Path(root)
    try:
        rel_parent = path.parent.relative_to(root)
    except ValueError:
        return path.parent.name if path.parent.name else "."

    rel_str = rel_parent.as_posix()
    return rel_str if rel_str else "."


def ensure_connected_graph(
    G: nx.Graph,
    *,
    largest_component_only: bool = False,
    source: str | Path | None = None,
) -> nx.Graph:
    if G.number_of_nodes() == 0:
        raise ValueError(f"Graph is empty: {source or '<unknown>'}")

    if nx.is_connected(G):
        return G

    if not largest_component_only:
        raise ValueError(f"Graph is disconnected: {source or '<unknown>'}")

    largest_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_nodes).copy()


def load_graph_file(
    path: str | Path,
    *,
    root: str | Path | None = None,
    require_connected: bool = True,
    largest_component_only: bool = False,
    normalize_labels_flag: bool = False,
) -> GraphInstance:
    path = Path(path)
    G = load_adjlist_graph(path)

    if require_connected or largest_component_only:
        G = ensure_connected_graph(
            G,
            largest_component_only=largest_component_only,
            source=path,
        )

    if normalize_labels_flag:
        G = normalize_graph_labels(G)

    return GraphInstance(
        graph=G,
        graph_label=graph_name_from_path(path),
        graph_family=graph_family_from_path(path, root=root),
        source_path=str(path.resolve()),
    )


def iter_graph_files(
    root: str | Path,
    *,
    recursive: bool = True,
    suffixes: tuple[str, ...] = _DEFAULT_SUFFIXES,
) -> Iterable[Path]:
    root = Path(root)
    candidates = root.rglob("*") if recursive else root.glob("*")
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


def load_graph_folder(
    root: str | Path,
    *,
    recursive: bool = True,
    suffixes: tuple[str, ...] = _DEFAULT_SUFFIXES,
    require_connected: bool = True,
    largest_component_only: bool = False,
    normalize_labels_flag: bool = False,
) -> list[GraphInstance]:
    root = Path(root)
    instances: list[GraphInstance] = []
    for path in iter_graph_files(root, recursive=recursive, suffixes=suffixes):
        instance = load_graph_file(
            path,
            root=root,
            require_connected=require_connected,
            largest_component_only=largest_component_only,
            normalize_labels_flag=normalize_labels_flag,
        )
        instances.append(instance)
    return instances


def describe_instance(instance: GraphInstance) -> str:
    G = instance.graph
    return f"{instance.graph_family}/{instance.graph_label}: n={G.number_of_nodes()}, m={G.number_of_edges()}"


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "data"
    if root.exists():
        instances = load_graph_folder(root, recursive=True, require_connected=False)
        print(f"Project root: {project_root}")
        print(f"Data root: {root}")
        print(f"Found {len(instances)} candidate graph files under {root}")
        for inst in instances:
            print(describe_instance(inst))
    else:
        print(f"Path does not exist: {root}")
