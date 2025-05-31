# Neural Graph Memory Evaluation Tests

This folder contains 500 evaluation test cases used to validate the performance, recall mechanisms, and graph dynamics of the Neural Graph Memory system.

## Structure

Each test file is a `.json` file that simulates a multimodal memory entry and a query scenario. These include combinations of:
- Textual prompts
- Visual embeddings (represented symbolically)
- Temporal or causal sequences
- Expected recalled memories and graph structure

## Usage

To run these tests with the Neural Graph Memory system:

1. Clone the repository:
```bash
git clone https://github.com/StuckInTheNet/neural-graph-memory.git
cd neural-graph-memory
