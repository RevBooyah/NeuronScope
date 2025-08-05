# NeuronScope Pruning Analysis Guide

This guide explains how to use NeuronScope's pruning analysis features to identify and evaluate neurons for pruning in transformer models.

## Overview

The pruning analysis module provides comprehensive tools for:
- **Weight Analysis**: Identifying neurons with low weight magnitudes
- **Pruning Candidate Detection**: Finding neurons safe to prune
- **Impact Analysis**: Simulating pruning effects on model behavior
- **Importance Assessment**: Determining critical neurons to preserve

## Key Features

### 1. Weight Analysis
Analyzes model weights to identify patterns and sparsity:
- Overall model sparsity statistics
- Layer-wise weight magnitude distributions
- Identification of neurons with weights close to zero

### 2. Pruning Candidate Identification
Automatically identifies neurons that are good candidates for pruning:
- Configurable threshold percentiles (default: 10%)
- Weight magnitude ranking within layers
- Pruning safety scores

### 3. Pruning Impact Simulation
Simulates the effects of pruning specific neurons:
- Measures activation changes before/after pruning
- Calculates impact scores for individual neurons
- Identifies safe vs. risky pruning operations

### 4. Neuron Importance Analysis
Evaluates the importance of neurons in specific layers:
- Tests neurons with multiple input texts
- Calculates activation magnitude and variance
- Identifies critical neurons that should not be pruned

### 5. Batch Analysis
Efficiently analyzes multiple pruning candidates:
- Individual impact assessment
- Cumulative impact analysis
- Automated recommendations

## API Endpoints

### Weight Analysis
```bash
# Get comprehensive weight analysis
GET /api/pruning/weight-analysis

# Get pruning candidates with configurable threshold
GET /api/pruning/candidates?threshold=10.0
```

### Impact Analysis
```bash
# Analyze impact of pruning specific neurons
POST /api/pruning/impact-analysis
{
  "layer_index": 0,
  "neuron_indices": [0, 1, 2],
  "input_text": "Hello world, this is a test."
}

# Analyze neuron importance in a layer
POST /api/pruning/neuron-importance
{
  "layer_index": 0,
  "input_texts": [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is fascinating"
  ]
}
```

### Batch Analysis
```bash
# Perform batch analysis of multiple candidates
POST /api/pruning/batch-analysis
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

### Export
```bash
# Export analysis results to JSON
POST /api/pruning/export
{
  "type": "weight"  # or "impact"
}
```

## Usage Examples

### Python Backend Usage

```python
from src.backend.models.gpt2_loader import MultiModelLoader
from src.backend.pruning import WeightAnalyzer, PruningImpactAnalyzer

# Load model
model_loader = MultiModelLoader()
model, tokenizer = model_loader.load_model('gpt2')

# Initialize analyzers
weight_analyzer = WeightAnalyzer(model)
pruning_analyzer = PruningImpactAnalyzer(model, tokenizer)

# Get weight analysis
sparsity_analysis = weight_analyzer.get_sparsity_analysis()
print(f"Overall sparsity: {sparsity_analysis['overall_sparsity']:.3f}")

# Identify pruning candidates
candidates = weight_analyzer.identify_pruning_candidates(threshold_percentile=10.0)
print(f"Found {len(candidates)} pruning candidates")

# Analyze impact of pruning
impact = pruning_analyzer.simulate_neuron_pruning(
    layer_index=0, 
    neuron_indices=[0, 1, 2], 
    input_text="Hello world"
)
print(f"Impact score: {impact.impact_score:.3f}")
print(f"Safe to prune: {impact.impact_score < 0.1}")
```

### JavaScript Frontend Usage

```javascript
// Get weight analysis
const weightAnalysis = await fetch('/api/pruning/weight-analysis')
  .then(response => response.json());

console.log(`Overall sparsity: ${weightAnalysis.overall_sparsity}`);

// Get pruning candidates
const candidates = await fetch('/api/pruning/candidates?threshold=10.0')
  .then(response => response.json());

console.log(`Found ${candidates.total_candidates} candidates`);

// Analyze pruning impact
const impact = await fetch('/api/pruning/impact-analysis', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    layer_index: 0,
    neuron_indices: [0, 1, 2],
    input_text: "Hello world"
  })
}).then(response => response.json());

console.log(`Impact score: ${impact.impact_score}`);
console.log(`Safe to prune: ${impact.safe_to_prune}`);
```

## Pruning Workflow

### 1. Initial Analysis
```python
# Get overall model statistics
sparsity_analysis = weight_analyzer.get_sparsity_analysis()
print(f"Model has {sparsity_analysis['total_parameters']:,} parameters")
print(f"Current sparsity: {sparsity_analysis['overall_sparsity']:.3f}")
```

### 2. Identify Candidates
```python
# Find neurons with low weight magnitudes
candidates = weight_analyzer.identify_pruning_candidates(threshold_percentile=10.0)
safe_candidates = [c for c in candidates if c.is_pruning_candidate]
print(f"Identified {len(safe_candidates)} safe pruning candidates")
```

### 3. Impact Assessment
```python
# Test impact of pruning individual neurons
for candidate in safe_candidates[:5]:  # Test first 5
    impact = pruning_analyzer.simulate_neuron_pruning(
        candidate.layer_index,
        [candidate.neuron_index],
        "Hello world"
    )
    print(f"Neuron {candidate.neuron_index}: impact={impact.impact_score:.3f}")
```

### 4. Batch Testing
```python
# Test cumulative impact of pruning multiple neurons
batch_candidates = [
    {"layer_index": c.layer_index, "neuron_index": c.neuron_index}
    for c in safe_candidates[:10]
]

batch_results = pruning_analyzer.batch_pruning_analysis(
    ["Hello world", "Test input"],
    batch_candidates
)

print(f"Safe to prune: {batch_results['recommendations']['safe_to_prune']}")
print(f"Risky to prune: {batch_results['recommendations']['risky_to_prune']}")
```

### 5. Export Results
```python
# Save analysis for later reference
weight_analyzer.export_weight_analysis("pruning_analysis.json")
```

## Interpretation Guidelines

### Impact Score Thresholds
- **< 0.1**: Safe to prune (minimal impact)
- **0.1 - 0.3**: Moderate impact (prune with caution)
- **> 0.3**: High impact (avoid pruning)

### Weight Magnitude Guidelines
- **< 0.01**: Very low weight (good pruning candidate)
- **0.01 - 0.05**: Low weight (potential candidate)
- **> 0.05**: Normal weight (preserve)

### Sparsity Targets
- **< 5%**: Low sparsity (minimal pruning opportunities)
- **5-20%**: Moderate sparsity (good pruning potential)
- **> 20%**: High sparsity (extensive pruning possible)

## Best Practices

### 1. Start Conservative
- Begin with low threshold percentiles (5-10%)
- Test individual neurons before batch pruning
- Monitor impact scores carefully

### 2. Use Multiple Test Inputs
- Test with diverse input texts
- Include domain-specific examples
- Consider edge cases and rare inputs

### 3. Layer-Specific Analysis
- Different layers may have different pruning characteristics
- Analyze each layer separately
- Consider layer depth and function

### 4. Iterative Approach
- Prune in small batches
- Re-evaluate after each pruning step
- Monitor model performance metrics

### 5. Backup and Validation
- Export analysis results before pruning
- Keep original model for comparison
- Validate pruned model on test data

## Troubleshooting

### Common Issues

**High Impact Scores**: If many neurons show high impact scores:
- Lower the threshold percentile
- Test with more diverse inputs
- Consider layer-specific analysis

**Memory Issues**: For large models:
- Use quantization
- Analyze layers individually
- Reduce batch sizes

**Inconsistent Results**: If results vary significantly:
- Use more test inputs
- Increase the number of samples
- Check for model loading issues

### Performance Optimization

- Use GPU acceleration when available
- Cache analysis results
- Parallelize batch analysis
- Use approximate methods for large models

## Future Enhancements

Planned features for future releases:
- Interactive pruning candidate selection UI
- Real-time pruning simulation
- Integration with fine-tuning workflows
- Advanced sparsity pattern analysis
- Cross-model pruning comparison
- Automated pruning strategy optimization

## Support

For issues or questions about pruning analysis:
1. Check the test script: `python test_pruning_analysis.py`
2. Review API documentation
3. Examine example outputs in the data directory
4. Consult the main NeuronScope documentation 