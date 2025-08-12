
import json
import os

def load_graph_data(base_path='data'):
    """
    Loads synthetic graph data including nodes, edges, queries, and annotations.

    Args:
        base_path (str): Path to the directory containing the data files.

    Returns:
        dict: A dictionary with keys 'nodes', 'edges', 'queries', and 'annotations'.
    """
    graph_path = os.path.join(base_path, 'toy_graph.json')
    queries_path = os.path.join(base_path, 'synthetic_queries.json')
    annotations_path = os.path.join(base_path, 'annotations.json')

    with open(graph_path, 'r') as f:
        graph = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    return {
        "nodes": graph.get("nodes", []),
        "edges": graph.get("edges", []),
        "queries": queries,
        "annotations": annotations
    }

if __name__ == "__main__":
    data = load_graph_data()
    print(f"Loaded {len(data['nodes'])} nodes, {len(data['edges'])} edges")
    print(f"Loaded {len(data['queries'])} queries and {len(data['annotations'])} annotations")
