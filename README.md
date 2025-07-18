# Neural Graph Memory (NGM)

**A Biologically-Inspired Graph Architecture for Multimodal Episodic Recall in AI Systems**

[![arXiv](https://img.shields.io/badge/arXiv-2507.12345-b31b1b.svg)](https://arxiv.org/abs/2507.12345)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StuckInTheNet/neural-graph-memory/blob/main/notebooks/NGM_Demo__.ipynb)

Neural Graph Memory (NGM) is a memory system for AI agents that combines graph-based structure, CLIP-powered multimodal embeddings, and contextualized episodic recall. Designed for scenarios that demand long-range retrieval, temporal continuity, and associative linking across modalities.

## Overview

NGM introduces a biologically-motivated memory system:

- **Graph-Structured Memory:** Nodes represent multimodal events (text, image, audio, timestamp); edges encode relationships such as temporal succession, semantic similarity, and co-location.
- **CLIP-Powered Embeddings:** We leverage OpenAI’s CLIP model to unify visual and textual modalities for retrieval.
- **Query-Aware Traversal:** Memory retrieval is implemented via graph traversal and embedding similarity.

##  Project Structure

```bash
neural-graph-memory/
├── notebooks/
│   └── NGM_Demo.ipynb              # Colab demo notebook
├── src/
│   ├── memory.py                   # Core graph memory logic
│   └── load_data.py                # Utility for loading synthetic memory data
├── data/
│   ├── toy_graph.json
│   ├── synthetic_queries.json
│   └── annotations.json
│   └── Synthetic_Graph_Nodes.csv
├── assets/                         # Sample images and diagrams
├── paper/
│   ├── Neural Graph Memory (NGM) -A Biologically-Inspired Graph Architecture for Multimodal Episodic Recall in AI Systems 
│   ├── Images
│   └── nodes.png
│   └── architecture.png
│   ├── references.md      
├── scripts/
│   └── benchmark_ngm.py            # Benchmarking logic
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
```

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/StuckInTheNet/neural-graph-memory.git
cd neural-graph-memory
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or in Colab: dependencies are installed automatically in the notebook.

### 3. Run the Demo Notebook

Open the notebook:

```bash
notebooks/NGM_Demo__.ipynb
```

Or run directly on [Google Colab](https://colab.research.google.com/github/StuckInTheNet/neural-graph-memory/blob/main/notebooks/NGM_Demo__.ipynb).

##  Benchmarking & Evaluation

Evaluate retrieval accuracy and traversal efficiency using:

```bash
python scripts/benchmark_ngm.py --data_dir=data --top_k=3
```

Sample output:

```
Recall@1: 0.76
Recall@3: 0.92
Graph traversal latency: 13.2ms
```

Optional visualization support coming soon.

## Data & Samples

Synthetic memory graphs are provided in the `data/` folder and include:

- `toy_graph.json`: multimodal memory nodes and relationships
- `synthetic_queries.json`: retrieval queries
- `annotations.json`: ground-truth relevance labels

Use `src/load_data.py` to load and manipulate them.

## Assets

All visual materials used in the notebook and paper are stored in `assets/`, including:

- Architecture diagrams
- Example memory stimuli (images)
- CLIP-compatible visuals

## Citation

If you use this code or refer to Neural Graph Memory in your work, please cite:

```bibtex
@article{fisher2025neuralgraphmemory,
  title={Neural Graph Memory: A Biologically Inspired Architecture for Multimodal Episodic Recall},
  author={Fisher, Matthew},
  journal={arXiv preprint arXiv:2507.12345},
  year={2025}
}
```

##  Related Work

- [CLIP: Learning Transferable Visual Models from Natural Language](https://openai.com/research/clip)
- [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903)
- [Neural Episodic Control](https://arxiv.org/abs/1703.01988)
- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)

##  Contact

Author: Matthew Fisher  
Email: [matthew.fisher@brown.edu](mailto:matthew.fisher@brown.edu)  
Website: [itsmefish.com](https://itsmefish.com)

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
