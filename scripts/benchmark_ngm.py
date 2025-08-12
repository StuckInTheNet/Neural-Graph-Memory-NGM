import os
import sys
import time
import json

# Allow imports from SRC/ (note: directory is capitalized)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SRC'))

from load_data import load_graph_data
from memory import NeuralGraphMemory

def main():
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data = load_graph_data(data_dir)

    nodes = data['nodes']
    edges = data['edges']
    queries = data['queries']
    annotations = data['annotations']

    # Initialize memory
    ngm = NeuralGraphMemory()
    ngm.add_nodes(nodes)
    ngm.add_edges(edges)

    print(f"[INFO] Loaded {len(nodes)} nodes, {len(edges)} edges, and {len(queries)} queries.")

    # Run benchmark
    total_time = 0.0
    correct = 0
    results = []

    for i, query in enumerate(queries):
        start = time.time()
        # Use the correct key from query data
        query_text = query.get('query', query.get('input', ''))
        answer = ngm.query(query_text)
        duration = time.time() - start
        total_time += duration

        # Get ground truth annotation
        expected = annotations[i]['expected']
        is_correct = expected.lower() in answer.lower()
        results.append((query_text, answer, expected, is_correct))

        if is_correct:
            correct += 1

    # Report
    print("\n[RESULTS]")
    for q, a, e, ok in results:
        status = "✓" if ok else "✗"
        print(f"{status} Q: {q}\n   A: {a}\n   Expected: {e}\n")

    print(f"Total queries: {len(queries)}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {correct / len(queries) * 100:.2f}%")
    print(f"Avg retrieval time: {total_time / len(queries):.4f} sec")

if __name__ == "__main__":
    main()
