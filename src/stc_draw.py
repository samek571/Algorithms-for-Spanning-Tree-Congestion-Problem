from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt
from stc_core import validate_tree, canon_edge, CongestionResult


#drawing helper
def _default_draw_pos(G: nx.Graph, seed: int = 0):
    """
    Choose a drawing layout.

    Preference:
    1. existing node attribute "pos"
    2. graphviz layout if available
    3. spring layout
    """
    pos_attr = nx.get_node_attributes(G, "pos")
    if len(pos_attr) == G.number_of_nodes():
        return pos_attr

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(G, prog="neato")
    except Exception:
        return nx.spring_layout(G, seed=seed)

#drawing helper
def draw_graph(
        G: nx.Graph,
        *,
        pos=None,
        title: str | None = None,
        node_size: int = 700,
        with_labels: bool = True,
        save_path: str | Path | None = None,
        show: bool = True,
        seed: int = 0,
):
    if pos is None:
        pos = _default_draw_pos(G, seed=seed)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, pos=pos, with_labels=with_labels, node_size=node_size, edge_color="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return pos

#drawing helper
def draw_tree(
        T: nx.Graph,
        congestion: CongestionResult | None = None,
        *,
        pos=None,
        title: str | None = None,
        node_size: int = 700,
        with_labels: bool = True,
        save_path: str | Path | None = None,
        show: bool = True,
        seed: int = 0,
):
    if pos is None:
        pos = _default_draw_pos(T, seed=seed)

    nx.draw_networkx(T, pos=pos, with_labels=with_labels, node_size=node_size, edge_color="tab:red", width=2.5)

    if congestion is not None:
        edge_labels = {}
        for u, v in T.edges():
            edge_labels[(u, v)] = congestion.edge_congestion[canon_edge(u, v)]
        nx.draw_networkx_edge_labels(T, pos=pos, edge_labels=edge_labels, font_size=9)

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return pos

#drawing master
def draw_graph_with_tree(
        G: nx.Graph,
        T: nx.Graph,
        congestion: CongestionResult | None = None,
        *,
        pos=None,
        title: str | None = None,
        node_size: int = 700,
        with_labels: bool = True,
        show_non_tree_edges: bool = True,
        highlight_worst_edges: bool = True,
        save_path: str | Path | None = None,
        show: bool = True,
        seed: int = 0,
):
    """
    - non-tree edges: light gray dashed
    - tree edges: red, thicker
    - worst congestion tree edges: orange, thickest
    - optional congestion labels on tree edges
    """
    plt.figure(figsize=(9, 7))
    validate_tree(G, T)

    if pos is None:
        pos = _default_draw_pos(G, seed=seed)

    tree_edges = {canon_edge(u, v) for u, v in T.edges()}
    non_tree_edges = []
    for u, v in G.edges():
        if canon_edge(u, v) not in tree_edges:
            non_tree_edges.append((u, v))

    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size)
    if with_labels:
        nx.draw_networkx_labels(G, pos=pos)

    if show_non_tree_edges and non_tree_edges:
        nx.draw_networkx_edges(G, pos=pos, edgelist=non_tree_edges, edge_color="lightgray", style="dashed", width=1.21, alpha=0.8)

    nx.draw_networkx_edges(T, pos=pos, edgelist=list(T.edges()), edge_color="tab:red", width=2.6)

    if congestion is not None:
        edge_labels = {}
        worst = []
        for u, v in T.edges():
            value = congestion.edge_congestion[canon_edge(u, v)]
            edge_labels[(u, v)] = value
            if value == congestion.max_congestion:
                worst.append((u, v))

        nx.draw_networkx_edge_labels(T, pos=pos, edge_labels=edge_labels, font_size=9)

        if highlight_worst_edges and worst:
            nx.draw_networkx_edges(T, pos=pos, edgelist=worst, edge_color="orange", width=4.0)

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return pos
