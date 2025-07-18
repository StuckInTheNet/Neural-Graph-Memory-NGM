
# 🧠 Neural Graph Memory (NGM)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/downloads/)
[![ArXiv](https://img.shields.io/badge/arXiv-Preprint-lightgrey.svg)](https://arxiv.org/abs/placeholder)


A biologically inspired graph-structured memory architecture for long-term multimodal recall in AI agents.

Neural Graph Memory (NGM) enables structured memory storage and retrieval using node-based representations, associative links, and multimodal embeddings. Designed for agents that operate in complex environments, NGM supports episodic recall, long-range reasoning, and retrieval across time, space, and modality.


## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/StuckInTheNet/neural-graph-memory.git
cd neural-graph-memory
pip install -r requirements.txt
```

## Example Usage

Run the demo in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StuckInTheNet/neural-graph-memory-Work-In-Progress-/blob/main/NGM_Demo__.ipynb)

Or run locally
git clone https://github.com/StuckInTheNet/neural-graph-memory-Work-In-Progress-.git
cd neural-graph-memory-Work-In-Progress-
pip install -r requirements.txt
jupyter notebook NGM_Demo__.ipynb


# Synthetic Graph Nodes

Here's a synthetic dataset of memory graph nodes, generated for use in the NGM experiments. These nodes simulate multimodal entries (image, text, audio, event) along with example queries and annotations.

## Files

- `toy_graph.json`: Core graph node and edge data used to simulate memory storage structure.
- `synthetic_queries.json`: Sample user queries for retrieval tasks across memory nodes.
- `annotations.json`: Ground truth labels or expected responses used for evaluation and benchmarking.

## Format

All files are in JSON format and can be loaded with `load_data.py` using:

```python
from src.load_data import load_graph_data
data = load_graph_data("data/")


### Examples Included

- **Image + Caption** – Visual memory with annotation  
- **Text + Audio** – Simulated voice note transcription  
- **Image + Timestamp + Context** – Episodic memory with temporal context  
- **Diagram + Interpretation** – Technical memory with meaning and structure


## Paper

Neural Graph Memory: A Biologically Inspired Graph Architecture for Multimodal Episodic Recall in AI Systems  
Matthew Fisher, 2025

 [Read the Paper (arXiv preprint)](https://arxiv.org/abs/placeholder)  
Included in this repo: `ngm_arxiv_final_pristine.tex` and `references_ngm.bib`


## Directory Structure

```
neural-graph-memory/
│
├── src/
│   └── memory.py       ← Core memory logic (NGM class, embedding, retrieval)
│
├── assets/             ← Images and diagrams
│
├── NGM_Demo__.ipynb    ← Clean notebook that imports from src.memory
├── README.md
├── requirements.txt
└── LICENSE

```

## License

MIT License. See `LICENSE` file for details.

## Citation

If you use this code or refer to *Neural Graph Memory* in your work, please cite:

```bibtex
@article{fisher2025neuralgraphmemory,
  title={Neural Graph Memory: A Biologically Inspired Architecture for Multimodal Episodic Recall},
  author={Fisher, Matthew},
  journal={arXiv preprint arXiv:2507.12345},
  year={2025}
}
