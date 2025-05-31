# Neural Graph Memory

Can an AI remember the world like we do?

This repo is an early stage experiment in building a more 'human like' memory system for AI, one that isn’t just a giant token buffer. Instead of stuffing everything into a fixed length context window, we store memory as a graph: events become nodes, relationships form edges, and recall happens through semantic and contextual similarity. It's built around the idea of biomimicry, borrowing inspiration from how the brain handles memory episodic, sensory, and associative.

This is a working prototype. It encodes multimodal experiences (like “coffee smell” in the “morning kitchen scene”), builds a memory graph from them, and uses that graph to answer recall prompts. It’s lightweight, extensible, and designed for experimentation.

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/neural-graph-memory/blob/main/notebooks/demo.ipynb)

---

## What This Is

- A graph memory engine for multimodal AI recall
- Inspired by hippocampal indexing, episodic memory, and Hebbian association
- Built in Python using NetworkX and SentenceTransformers
- Includes benchmark tasks, ablations, and a graph visualizer

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/neural-graph-memory.git
cd neural-graph-memory
pip install -r requirements.txt
python memory_graph.py
```

Example output:
```
Query: 'coffee smell' → Recalled: 'morning kitchen scene'
Recall Accuracy: 89.4%
```

## What's Inside

| File              | Purpose                                            |
|-------------------|----------------------------------------------------|
| `memory_graph.py` | Build the memory graph, encode sensory events      |
| `evaluate.py`     | Run benchmarks and ablations                       |
| `visualize.py`    | Plot the memory graph for inspection               |
| `test_cases.json` | 500+ query context pairs for recall evaluation     |
| `notebooks/demo.ipynb` | Walkthrough with visual examples              |
| `requirements.txt`| Dependency list for Python environment setup       |

## Why It Matters

- Memory isn’t flat. This stores it as a graph with real structure
- Recall is contextual and semantic, not just string matching
- Nodes can represent sight, sound, smell, etc.
- Edges track time, co-occurrence, causality, and more
- Inspired by actual neuroscience, not just prompt hacks

## Benchmarks

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 89.4%   |
| Precision | 87.8%   |
| Recall    | 90.3%   |

Performance is consistent and holds up under ablation.

## Paper

This repo supports the preprint:

**“Toward Human-Like Memory in AI: A Bio Inspired Graph Architecture for Multimodal Reasoning”**  
Matt Fisher, 2024

## Future Work

- Replace text based cues with actual sensory inputs (CLIP, Whisper, IoT)
- Tie into agent systems for grounded long term memory
- Add pruning, decay, and novelty based updates

## Contact

Built by [Matt Fisher](https://www.linkedin.com/in/itsmefish/)  
📧 Fish@itsmefish.com

## License

MIT License. Fork it, remix it, or use it to make your system a little smarter.
