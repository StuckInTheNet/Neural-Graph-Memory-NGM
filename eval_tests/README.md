
# Evaluation Tests for Neural Graph Memory

This directory contains a suite of 500 evaluation test cases used to validate the behavior, robustness, and retrieval accuracy of the Neural Graph Memory (NGM) system. Each test case includes a query and expected memory outputs to simulate real-world reasoning tasks.

## Structure

Each test case is a `.json` file following a consistent schema:

```json
{
  "id": "test_001",
  "query": "Where did I leave my keys last night?",
  "expected_context": [
    "You came home around 9 PM.",
    "You placed your keys on the kitchen counter."
  ],
  "reasoning_type": "episodic",
  "modality": ["text"]
}
```

- **id**: Unique identifier for the test case.
- **query**: The input question or prompt to the NGM system.
- **expected_context**: The relevant memory or set of memories expected to be recalled.
- **reasoning_type**: Category of cognitive process (e.g., episodic, causal, semantic).
- **modality**: Indicates the input/output media involved in the test (e.g., text, image, audio).

## Usage

To run the evaluation tests against your model implementation:

1. Ensure your model exposes an API or CLI to input a query and return a set of memory responses.
2. Iterate through the JSON files, passing the `query` field into your system.
3. Compare your system's output with the `expected_context`.

## Schema Compliance

All files in this folder should comply with the above JSON structure. Schema validation can be performed using standard tools such as `jsonschema`.

## Licensing

These tests are released under the MIT License. See `LICENSE` for details.
