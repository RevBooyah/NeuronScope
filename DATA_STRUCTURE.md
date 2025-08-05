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

---

## 5. Weight Analysis Output

**File:** `pruning/weight_analysis.json`

```json
{
  "sparsity_analysis": {
    "overall_sparsity": 0.023,
    "total_parameters": 124439808,
    "non_zero_parameters": 121539456,
    "layer_sparsity": {
      "transformer.h.0.mlp.c_fc": {
        "sparsity": 0.0,
        "total_params": 589824,
        "non_zero_params": 589824
      }
    },
    "layer_stats": [
      {
        "layer_name": "transformer.h.0.mlp.c_fc",
        "layer_index": 0,
        "total_parameters": 589824,
        "non_zero_parameters": 589824,
        "sparsity": 0.0,
        "mean_magnitude": 0.045,
        "std_magnitude": 0.032,
        "min_magnitude": 0.0,
        "max_magnitude": 0.234,
        "l1_norm": 26542.08,
        "l2_norm": 1234.56
      }
    ]
  },
  "pruning_candidates": [
    {
      "layer_index": 0,
      "neuron_index": 42,
      "weight_magnitude": 0.012,
      "weight_rank": 8,
      "is_pruning_candidate": true,
      "pruning_score": 0.89
    }
  ]
}
```

- `sparsity_analysis`: Overall and layer-wise sparsity statistics
- `pruning_candidates`: Neurons identified as good pruning candidates
- `pruning_score`: Higher values indicate neurons more likely to be safely pruned

---

## 6. Pruning Impact Analysis Output

**File:** `pruning/impact_analysis.json`

```json
{
  "individual_impacts": [
    {
      "layer_index": 0,
      "neuron_index": 42,
      "impact_score": 0.023,
      "mean_change": 0.015,
      "max_change": 0.089,
      "safe_to_prune": true
    }
  ],
  "cumulative_impact": {
    "impact_score": 0.156,
    "mean_change": 0.089,
    "max_change": 0.234
  },
  "recommendations": {
    "safe_to_prune": 15,
    "risky_to_prune": 3,
    "suggested_batch_size": 10,
    "high_impact_neurons": [
      {
        "layer_index": 5,
        "neuron_index": 128,
        "impact_score": 0.567,
        "mean_change": 0.234,
        "max_change": 0.789,
        "safe_to_prune": false
      }
    ]
  }
}
```

- `individual_impacts`: Impact analysis for each pruned neuron
- `cumulative_impact`: Combined impact of pruning multiple neurons
- `recommendations`: Automated pruning recommendations and safety assessments