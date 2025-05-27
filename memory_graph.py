import networkx as nx
import numpy as np

def build_memory_graph(model):
    G = nx.Graph()

    def add(node_id, content, context):
        embedding = model.encode(content + ' ' + context)
        G.add_node(node_id, embedding=embedding, content=content, context=context)

    add(0, "fresh coffee aroma", "morning kitchen scene")
    add(1, "sunlight through window", "morning kitchen scene")
    add(2, "sound of birds chirping", "morning backyard")
    add(3, "tea kettle whistling", "afternoon kitchen")
    add(4, "rain tapping on glass", "evening porch")

    G.add_edge(0, 1, relation="temporal")
    G.add_edge(1, 2, relation="temporal")
    G.add_edge(2, 3, relation="causal")
    G.add_edge(3, 4, relation="temporal")

    return G

def recall_memory(graph, cue, model):
    cue_embedding = model.encode(cue)
    similarities = []

    for node_id, data in graph.nodes(data=True):
        similarity = np.dot(cue_embedding, data["embedding"])
        similarities.append((similarity, node_id))

    similarities.sort(reverse=True)
    top_node = similarities[0][1]
    return graph.nodes[top_node]
