# NeuronScope Analysis Log

This file is a working journal of observed phenomena, findings, and patterns discovered through visualizing neuron activations in transformer models.

Use this file to capture:
- Unexpected or interesting neuron behaviors
- Activation patterns across prompts
- Notable clustering/grouping trends
- Drift behaviors across fine-tuned checkpoints

---

## 📍 Entry Format Template

### Date:
YYYY-MM-DD

### Prompt:
"The quick brown fox jumps..."

### Observation:
Layer 8, Neuron 512 consistently shows high activation on verbs across multiple prompts.

### Interpretation:
This neuron may encode predicate structures or verb emphasis.

### Follow-up:
Add to list of tracked polysemantic candidates.

---

## Example Entry

### Date:
2025-07-30

### Prompt:
"Translate the following sentence to French: The cat sat on the mat."

### Observation:
Layer 10, Neuron 117 spikes strongly on the word "Translate" and "sat" but not on the noun tokens.

### Interpretation:
This neuron may specialize in action-command contexts and simple past tense predicates.

### Follow-up:
Compare activation across multiple "Translate" prompts.

---