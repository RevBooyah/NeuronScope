# NeuronScope API Documentation

This document provides comprehensive documentation for all NeuronScope API endpoints, including the new pruning analysis features.

## Base URL

All API endpoints are available at: `http://localhost:5001/api/`

## Authentication

Currently, no authentication is required for local development. All endpoints are publicly accessible.

## Response Format

All API responses are returned in JSON format with the following structure:

```json
{
  "status": "success|error",
  "data": { ... },
  "message": "Optional description"
}
```

## Health Check

### GET /api/health

Check if the API server is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "NeuronScope API server is running"
}
```

---

## Pruning Analysis Endpoints

### GET /api/pruning/weight-analysis

Get comprehensive weight analysis for the current model.

**Description:** Analyzes model weights to identify patterns, sparsity, and weight magnitude distributions across all layers.

**Response:**
```json
{
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
}
```

**Performance Note:** This endpoint may take 30-60 seconds for large models like GPT-2 due to the computational complexity of analyzing all weights.

---

### GET /api/pruning/candidates

Get pruning candidates for the current model.

**Description:** Identifies neurons that are good candidates for pruning based on weight magnitude analysis.

**Query Parameters:**
- `threshold` (float, optional): Percentile threshold for identifying candidates (default: 10.0)

**Example Request:**
```
GET /api/pruning/candidates?threshold=10.0
```

**Response:**
```json
{
  "candidates": [
    {
      "layer_index": 0,
      "neuron_index": 42,
      "weight_magnitude": 0.012,
      "weight_rank": 8,
      "is_pruning_candidate": true,
      "pruning_score": 0.89
    }
  ],
  "total_candidates": 64512,
  "threshold_percentile": 10.0
}
```

**Field Descriptions:**
- `layer_index`: Index of the transformer layer
- `neuron_index`: Index of the neuron within the layer
- `weight_magnitude`: Average magnitude of the neuron's weights
- `weight_rank`: Rank of the neuron's weight magnitude within its layer
- `is_pruning_candidate`: Whether this neuron is identified as a pruning candidate
- `pruning_score`: Score indicating how safe it is to prune (higher = safer)

**Performance Note:** This endpoint may take 30-60 seconds for large models.

---

### POST /api/pruning/impact-analysis

Analyze the impact of pruning specific neurons.

**Description:** Simulates pruning specific neurons and measures the impact on model activations.

**Request Body:**
```json
{
  "layer_index": 0,
  "neuron_indices": [0, 1, 2],
  "input_text": "Hello world, this is a test for pruning analysis."
}
```

**Field Descriptions:**
- `layer_index` (int, required): Index of the transformer layer
- `neuron_indices` (array of ints, required): List of neuron indices to simulate pruning
- `input_text` (string, required): Input text to test the pruning impact

**Response:**
```json
{
  "impact_score": 0.023,
  "mean_change": 0.015,
  "max_change": 0.089,
  "affected_neurons": [0, 1, 2],
  "safe_to_prune": true
}
```

**Field Descriptions:**
- `impact_score`: Normalized impact score (lower = safer to prune)
- `mean_change`: Average change in activations
- `max_change`: Maximum change in activations
- `affected_neurons`: List of neurons that were simulated as pruned
- `safe_to_prune`: Boolean indicating if pruning is considered safe (impact_score < 0.1)

**Impact Score Guidelines:**
- `< 0.1`: Safe to prune (minimal impact)
- `0.1 - 0.3`: Moderate impact (prune with caution)
- `> 0.3`: High impact (avoid pruning)

**Performance Note:** This endpoint may take 30-120 seconds depending on the number of neurons and input length.

---

### POST /api/pruning/neuron-importance

Analyze the importance of neurons in a specific layer.

**Description:** Evaluates the importance of all neurons in a specific layer by testing them with multiple input texts.

**Request Body:**
```json
{
  "layer_index": 0,
  "input_texts": [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is fascinating"
  ]
}
```

**Field Descriptions:**
- `layer_index` (int, required): Index of the transformer layer to analyze
- `input_texts` (array of strings, required): List of input texts to test with

**Response:**
```json
{
  "importance_scores": [
    {
      "layer_index": 0,
      "neuron_index": 42,
      "activation_magnitude": 0.156,
      "activation_variance": 0.089,
      "impact_score": 0.234,
      "is_critical": true
    }
  ],
  "total_neurons": 768,
  "critical_neurons": 192
}
```

**Field Descriptions:**
- `activation_magnitude`: Average activation magnitude across test inputs
- `activation_variance`: Variance of activations across test inputs
- `impact_score`: Impact score when this neuron is pruned
- `is_critical`: Whether this neuron is identified as critical (high impact when pruned)
- `total_neurons`: Total number of neurons in the layer
- `critical_neurons`: Number of neurons identified as critical

**Performance Note:** This endpoint may take 2-5 minutes as it tests each neuron individually.

---

### POST /api/pruning/batch-analysis

Perform batch analysis of multiple pruning candidates.

**Description:** Efficiently analyzes multiple pruning candidates and provides recommendations.

**Request Body:**
```json
{
  "pruning_candidates": [
    {"layer_index": 0, "neuron_index": 0},
    {"layer_index": 0, "neuron_index": 1},
    {"layer_index": 1, "neuron_index": 0}
  ],
  "input_texts": ["Hello world", "Test input"],
  "batch_size": 10
}
```

**Field Descriptions:**
- `pruning_candidates` (array of objects, required): List of neurons to analyze
- `input_texts` (array of strings, required): Input texts to test with
- `batch_size` (int, optional): Maximum batch size for cumulative analysis (default: 10)

**Response:**
```json
{
  "individual_impacts": [
    {
      "layer_index": 0,
      "neuron_index": 0,
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

**Field Descriptions:**
- `individual_impacts`: Impact analysis for each individual neuron
- `cumulative_impact`: Combined impact of pruning multiple neurons together
- `recommendations`: Automated pruning recommendations
  - `safe_to_prune`: Number of neurons safe to prune
  - `risky_to_prune`: Number of neurons risky to prune
  - `suggested_batch_size`: Recommended batch size for pruning
  - `high_impact_neurons`: List of neurons with high impact scores

**Performance Note:** This endpoint may take 1-3 minutes depending on the number of candidates.

---

### POST /api/pruning/export

Export pruning analysis results to JSON file.

**Description:** Saves analysis results to a JSON file in the data directory for later reference.

**Request Body:**
```json
{
  "type": "weight"
}
```

**Field Descriptions:**
- `type` (string, required): Type of analysis to export ("weight" or "impact")

**Response:**
```json
{
  "message": "Weight analysis exported successfully",
  "file_path": "/path/to/data/pruning/weight_analysis.json"
}
```

**Supported Export Types:**
- `"weight"`: Exports weight analysis and pruning candidates
- `"impact"`: Not yet implemented (planned for future)

---

## Model Management Endpoints

### GET /api/models

Get list of available models.

**Response:**
```json
{
  "models": {
    "gpt2": {
      "name": "gpt2",
      "family": "gpt2",
      "layers": 12,
      "hidden_size": 768,
      "description": "GPT-2 Small (124M parameters)",
      "recommended": true
    }
  }
}
```

---

### GET /api/models/{model_name}

Get detailed information about a specific model.

**Response:**
```json
{
  "name": "gpt2",
  "family": "gpt2",
  "layers": 12,
  "hidden_size": 768,
  "num_attention_heads": 12,
  "description": "GPT-2 Small (124M parameters)",
  "recommended": true,
  "size_category": "small"
}
```

---

### POST /api/models/switch

Switch to a different model.

**Request Body:**
```json
{
  "model_name": "gpt2-medium"
}
```

**Response:**
```json
{
  "message": "Successfully switched to gpt2-medium",
  "current_model": "gpt2-medium",
  "model_info": { ... },
  "memory_usage": { ... }
}
```

---

### GET /api/models/memory

Get current memory usage information.

**Response:**
```json
{
  "total_memory": 8589934592,
  "used_memory": 2147483648,
  "available_memory": 6442450944,
  "model_memory": 1073741824
}
```

---

## Activation Analysis Endpoints

### GET /api/activations/files

Get list of available activation files.

**Response:**
```json
{
  "files": ["hello_world.json", "test_prompt.json"],
  "count": 2
}
```

---

### GET /api/activations/{filename}

Get activation data for a specific file.

**Response:**
```json
{
  "prompt": "Hello world",
  "tokens": ["Hello", "world"],
  "layers": [
    {
      "layer_index": 0,
      "neurons": [
        {
          "neuron_index": 0,
          "activations": [0.132, 0.984]
        }
      ]
    }
  ]
}
```

---

### POST /api/activations/generate

Generate activation data for a new prompt.

**Request Body:**
```json
{
  "prompt": "Hello world",
  "model_name": "gpt2"
}
```

---

## Clustering Endpoints

### GET /api/clusters/files

Get list of available cluster files.

### GET /api/clusters/{filename}

Get cluster data for a specific file.

### POST /api/clusters/generate

Generate clustering analysis for activations.

**Request Body:**
```json
{
  "activation_file": "hello_world.json",
  "layer_index": 0,
  "n_clusters": 5
}
```

---

## Query Endpoints

### GET /api/queries/files

Get list of available query files.

### GET /api/queries/{filename}

Get query data for a specific file.

### POST /api/queries/neuron

Query for top tokens activating a specific neuron.

**Request Body:**
```json
{
  "layer_index": 0,
  "neuron_index": 42,
  "top_k": 10
}
```

### POST /api/queries/cluster

Query for top tokens activating a specific cluster.

**Request Body:**
```json
{
  "cluster_id": 0,
  "top_k": 10
}
```

---

## Sample Management Endpoints

### GET /api/samples

Get list of sample prompts.

**Response:**
```json
[
  "Hello world",
  "The quick brown fox jumps over the lazy dog",
  "Machine learning is fascinating"
]
```

---

### POST /api/samples

Add a new sample prompt.

**Request Body:**
```json
{
  "prompt": "New sample prompt"
}
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (authentication required)
- `403`: Forbidden (access denied)
- `404`: Not Found
- `500`: Internal Server Error

**Error Response Format:**
```json
{
  "error": "Error description",
  "details": "Additional error information"
}
```

---

## Performance Considerations

### Pruning Analysis Performance

The pruning analysis endpoints are computationally intensive and may take significant time:

- **Weight Analysis**: 30-60 seconds for GPT-2
- **Pruning Candidates**: 30-60 seconds for GPT-2
- **Impact Analysis**: 30-120 seconds per analysis
- **Neuron Importance**: 2-5 minutes per layer
- **Batch Analysis**: 1-3 minutes depending on batch size

### Optimization Tips

1. **Use smaller models** for faster analysis (GPT-2 small vs large)
2. **Limit input text length** for impact analysis
3. **Use smaller batch sizes** for batch analysis
4. **Cache results** by exporting analysis data
5. **Run analysis in background** for long-running operations

### Memory Usage

- GPT-2 Small: ~500MB RAM
- GPT-2 Medium: ~1GB RAM
- GPT-2 Large: ~2GB RAM
- LLaMA models: 4-8GB RAM depending on size

---

## Rate Limiting

Currently, no rate limiting is implemented. However, due to the computational intensity of pruning analysis, it's recommended to:

1. **Space out requests** to avoid overwhelming the server
2. **Use timeouts** in client applications
3. **Implement retry logic** for failed requests
4. **Monitor server resources** during heavy usage

---

## Development Notes

### Testing

Use the provided test scripts to verify functionality:

```bash
# Quick test (recommended)
python test_pruning_quick.py

# Full test (may take 10+ minutes)
python test_pruning_analysis.py
```

### Debugging

Enable debug logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Local Development

The API server runs on port 5001 by default. To change the port:

```bash
export PORT=5000
python run_api_server.py
```

---

## Future Enhancements

Planned API improvements:

1. **Async endpoints** for long-running operations
2. **Progress tracking** for batch operations
3. **Caching layer** for repeated requests
4. **Authentication system** for production use
5. **Rate limiting** for API protection
6. **WebSocket support** for real-time updates
7. **GraphQL interface** for complex queries 