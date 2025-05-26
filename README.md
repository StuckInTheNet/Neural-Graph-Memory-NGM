# Neural Graph Memory

_What if your AI could remember the world more like you do?_

This project is an experiment in building **human-like memory** for AI systems—drawing inspiration from the brain’s ability to store experiences through context, senses, and associations. Instead of stuffing everything into a linear context window, we use a **graph-based memory** model powered by bio-mimicry.

It’s not science fiction. It’s a practical prototype that uses modern embeddings and a custom memory graph to recall smells, sights, and sounds—from birds chirping in the backyard to the aroma of morning coffee in the kitchen.

 What This Is

- A **graph memory engine** for multi-modal AI recall
- Inspired by hippocampal indexing, episodic memory, and Hebbian learning
- Built in Python with NetworkX + SentenceTransformers
- Includes a demo, benchmark test set, and visualization tool

Quickstart

```bash
git clone https://github.com/mattfisherAI/neural-graph-memory.git
cd neural-graph-memory
pip install -r requirements.txt
python memory_graph.py
```

You’ll see something like:
```
Query: 'coffee smell' → Recalled: 'morning kitchen scene'
Recall Accuracy: 89.4%
```

---

## 🛠️ Repo Contents

| File              | Description                                      |
|-------------------|--------------------------------------------------|
| `memory_graph.py` | Build the graph, encode events, test recall      |
| `evaluate.py`     | Run 500 recall benchmarks + ablations           |
| `visualize.py`    | Plot the memory graph in 2D                     |
| `test_cases.json` | 500+ multi-modal cue/context pairs               |
| `requirements.txt`| Install NetworkX, transformers, numpy, etc.     |

---

##  Why It’s Interesting

- Stores memory **as a graph**, not a flat context window
- Recalls events using **semantic similarity**
- Adds **temporal and causal edges** between memories
- Simulates **smell, sound, sight** using embedding proxies
- Inspired by **neuroscience**, not just NLP

---

## What to Expect

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 89.4%   |
| Precision | 87.8%   |
| Recall    | 90.3%   |

The graph structure makes recall **context-sensitive**, **flexible**, and surprisingly robust.

---

## Paper

This repo accompanies the preprint:

> **“Toward Human-Like Memory in AI: A Bio-Inspired Graph Architecture for Multi-Modal Reasoning”**  
> Matt Fisher, 2024

---

## Future Plans

- Replace text proxies with real sensors (CLIP, Whisper, IoT)
- Integrate with LLM agents for memory-augmented conversations
- Add learning-based edge pruning and long-term storage strategies

---

## Contact

Created by [Matt Fisher](https://www.linkedin.com/in/itsmefish/)  
📍 Kirkland, WA  
📧 Fish@itsmefish.com

---

## License

MIT License. Fork it, remix it, use it in your next memory-enabled agent.
