import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize memory graph and embedding model
memory_graph = nx.Graph()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode sensory inputs and contexts into graph nodes
def add_memory_event(graph, sensory_input, context):
    embedding = model.encode(sensory_input + ' ' + context)
    node_id = len(graph)
    graph.add_node(node_id, embedding=embedding, content=sensory_input, context=context)
    return node_id

# Function to establish associations between events
def link_events(graph, node_a, node_b, relation="temporal"):
    graph.add_edge(node_a, node_b, relation=relation)

# Memory recall based on sensory cue
def recall_memory(graph, cue):
    cue_embedding = model.encode(cue)
    similarities = []
    for node_id, data in graph.nodes(data=True):
        similarity = np.dot(cue_embedding, data['embedding'])
        similarities.append((similarity, node_id))
    similarities.sort(reverse=True)
    most_similar = similarities[0][1]
    return graph.nodes[most_similar]

# Example implementation
node1 = add_memory_event(memory_graph, "fresh coffee aroma", "morning kitchen scene")
node2 = add_memory_event(memory_graph, "sunlight through window", "morning kitchen scene")
node3 = add_memory_event(memory_graph, "sound of birds chirping", "morning backyard")

link_events(memory_graph, node1, node2)
link_events(memory_graph, node2, node3)

# Recalling memory
def evaluate_recall_accuracy(graph, test_cases):
    correct = 0
    for cue, expected_context in test_cases:
        retrieved_memory = recall_memory(graph, cue)
        if retrieved_memory['context'] == expected_context:
            correct += 1
    accuracy = (correct / len(test_cases)) * 100
    return accuracy

# Example test cases for evaluation
test_cases = [
    ("coffee smell", "morning kitchen scene"),
    ("birds chirping", "morning backyard"),
    ("sunlight window", "morning kitchen scene")
]

accuracy = evaluate_recall_accuracy(memory_graph, test_cases)
print(f"Recall Accuracy: {accuracy}%")
