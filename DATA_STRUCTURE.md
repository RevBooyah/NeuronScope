# NeuronScope Data Structures

This document defines the structure of key data formats exchanged between the backend and frontend. All formats are JSON-based unless noted otherwise.

---

## 1. Neuron Activation Output

**File:** `activations/{prompt_name}.json`

```json
{
  "prompt": "Translate the sentence",
  "tokens": ["Translate", "the", "sentence"],
  "layers": [
    {
      "layer_index": 0,
      "neurons": [
        {
          "neuron_index": 0,
          "activations": [0.132, 0.984, -0.033]
        }
      ]
    }
  ]
}
```

- `tokens`: Tokenized form of the prompt.
- `activations`: One activation per token (same length as `tokens`).
- Shape: [layers][neurons][tokens].

---

## 2. Neuron Clustering Output

**File:** `clusters/{prompt_name}_clusters.json`

```json
{
  "layer": 10,
  "clusters": [
    {
      "cluster_id": 0,
      "neuron_indices": [3, 5, 12, 19],
      "centroid": [0.12, -0.04, 0.89]
    }
  ]
}
```

- `neuron_indices`: Neurons in this cluster (from this layer).
- `centroid`: Optional, for visualization.

---

## 3. Reverse Activation Query Output

**File:** `queries/{neuron_or_cluster_id}.json`

```json
{
  "query_type": "neuron",
  "neuron_index": 42,
  "top_tokens": [
    {"token": "the", "activation": 2.03},
    {"token": "to", "activation": 1.89}
  ]
}
```

- Sorted by strongest activation response.
- Used to build reverse-mapping visualizations.

---

## 4. Prompt Samples

**File:** `samples.json`

```json
[
  "Translate the sentence",
  "Hello world",
  "Summarize the following paragraph"
]
```

Used to preload sample visualizations for testing or UI demos.