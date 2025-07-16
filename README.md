# Neural Graph Memory (NGM)

> A Biologically-Inspired Graph Architecture for Long-Term Multimodal Episodic Memory in AI Systems  
> Developed by [Matthew Fisher](mailto:matthew_fisher@brown.edu)  
> 📄 [Read the Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX) (coming soon)

---

## Overview

**Neural Graph Memory (NGM)** is a structured memory architecture designed to extend long-term memory capabilities in AI agents. Inspired by hippocampal memory formation, NGM represents episodic events as dynamic, modality-aware graphs, enabling efficient retrieval, contextual reasoning, and temporal alignment across multimodal inputs.

NGM supports:

- Structured memory as sparse graphs with semantic nodes and typed edges
- Multimodal fusion (text, image, video, etc.)
- Associative and temporal recall via topological traversal
- Integration with transformer-based agents and LLMs

---

## Key Features

-  **Graph-based Memory** — Events encoded as graph nodes, linked by semantic, temporal, and contextual relationships.
-  **Modality-Aware Encoding** — Inputs from multiple modalities (e.g., language, vision) embedded into unified latent space.
-  **Retrieval by Traversal** — Fast nearest-neighbor + graph walk retrieval mechanisms, mimicking biological episodic recall.
-  **Benchmarked** — Outperforms baseline memory systems on multimodal and episodic tasks requiring long-range reasoning.

---

## 🛠️ Installation

```bash
git clone https://github.com/StuckInTheNet/neural-graph-memory.git
cd neural-graph-memory
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x required.

---

## 🧪 Reproducing Results

All experiments from the paper can be reproduced via:

```bash
python experiments/run_all.py
```

To run a specific benchmark:

```bash
python experiments/retrieve_temporal.py --config configs/temporal.yaml
```

Output logs and visualizations will be saved to `/results`.

---

## 📂 Repository Structure

```plaintext
📦 neural-graph-memory/
├── architecture/        # Core model definition (node, edge, graph classes)
├── encoders/            # Vision and language encoders (CLIP, BERT, etc.)
├── retriever/           # Associative retrieval and traversal engine
├── experiments/         # Evaluation scripts and benchmark configs
├── utils/               # Helper functions, logging, visualization
├── notebooks/           # Walkthrough examples (coming soon)
├── requirements.txt
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{fisher2025ngm,
  title={Neural Graph Memory: A Biologically-Inspired Graph Architecture for Multimodal Episodic Recall in AI Systems},
  author={Matthew Fisher},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Contributing

Contributions are welcome! Please open an issue or PR.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
