
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from memory_graph import build_memory_graph, recall_memory

def evaluate(graph, test_cases, model):
    correct = 0
    for case in test_cases:
        result = recall_memory(graph, case["cue"], model)
        match = result["context"] == case["expected_context"]
        print(f"Query: '{case['cue']}' → Recalled: '{result['context']}' | Expected: '{case['expected_context']}' | Match: {match}")
        if match:
            correct += 1
    accuracy = (correct / len(test_cases)) * 100
    print(f"\nRecall Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    with open("test_cases.json", "r") as f:
        test_cases = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    G = build_memory_graph(model)
    evaluate(G, test_cases, model)
