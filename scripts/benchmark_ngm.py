# scripts/benchmark_ngm.py

import time
import os
import sys

# Adjust the path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from load_data import load_graph_data
from memory import NeuralGraphMemory

def run_benchmark():
    print("🔍 Loading synthetic data...")
    data = load_graph_data('data')
    nodes = data['nodes']
    edges = data['edges']
    queries = data['queries']
    annotations = data['annotations']

    print("🧠 Initializing Neural Graph Memory...")
    ngm = NeuralGraphMemory()
    
    start_time = time.time()
    ngm.build_graph(nodes, edges)
    graph_build_time = time.time() - start_time
    print(f"✅ Graph build time: {graph_build_time:.4f} seconds")

    print("🔎 Running retrieval benchmark...")
    retrieval_times = []
    results = []

    for q in queries:
        start = time.time()
        retrieved = ngm.retrieve(q['query'])
        duration = time.time() - start
        retrieval_times.append(duration)

        results.append({
            "query": q['query'],
            "retrieved": retrieved,
            "expected": annotations.get(q['query'], 'N/A')
        })

    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    print(f"⏱ Average retrieval time per query: {avg_retrieval_time:.4f} seconds")

    print("\n📋 Sample Benchmark Results:")
    for r in results[:5]:  # Show top 5
        print(f"- Query: {r['query']}")
        print(f"  Retrieved: {r['retrieved']}")
        print(f"  Expected: {r['expected']}")
        print("")

if __name__ == "__main__":
    run_benchmark()
