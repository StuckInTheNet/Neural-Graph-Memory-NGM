import matplotlib.pyplot as plt
import networkx as nx
from memory_graph import build_memory_graph
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
G = build_memory_graph(model)

labels = {n: f"{d['content']}\n[{d['context']}]" for n, d in G.nodes(data=True)}
edge_labels = nx.get_edge_attributes(G, "relation")
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=2000, font_size=9)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Neural Graph Memory Visualization")
plt.axis("off")
plt.tight_layout()
plt.show()
