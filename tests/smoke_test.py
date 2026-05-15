import networkx as nx
from src.stc_arv import arv_balanced_cut

G = nx.cycle_graph(8)
side = arv_balanced_cut(G, seed=42)
print(f"Cut: {side}, size: {nx.cut_size(G, side, set(G.nodes()) - side)}")
