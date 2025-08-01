
# Cursor Project Guide: NeuronScope

This file is designed to guide AI development tools (like Cursor or Claude) to properly understand and implement the NeuronScope project.

---

## 🧠 Project Summary

NeuronScope is an open-source research and visualization platform for analyzing neuron activations in transformer-based language models. It enables forward (token → neuron) and reverse (neuron → token) analysis, bigram/n-gram activation tracing, and drift tracking across fine-tuning. It serves as a scientific instrumentation tool for LLM interpretability, neuron behavior mapping, and debugging.

Target users:
- AI/LLM researchers
- Transformer model developers
- Interpretability-focused engineers

---

## 🎯 MVP Feature Checklist

- [] CLI to load a HuggingFace model and run activation probes
- [] Forward activation tracing: input → activated neurons
- [] Inverse queries: neuron(s) → top activating tokens
- [] Bigram/n-gram token activation analysis
- [] Generate CSV and JSON outputs of activations
- [] Output static heatmaps (token vs. neuron) as PNG/SVG
- [] Simple Web UI for visualization (heatmaps, neuron views)
- [] Directory-based output storage per run

---

## 🚫 Non-Goals (for MVP)
Not going to be working on any of these features until expressly committed.

- Training or fine-tuning models
- Jupyter-only interfaces
- Deep model editing or patching
- Attention head analysis (for now)
- Use of proprietary APIs (e.g., OpenAI GPT-4)

---

## 🛠️ Tech Stack

- Python 3.10+
- HF `transformers`, `datasets`, `torch`
- Plotly / D3.js / Matplotlib (final choice pending)
- HTML/JS frontend served by `nscope serve`
- Optional inline C for acceleration and memory intensive actions
- Markdown + JSON output

---

## 📁 Data & File Conventions

### Input
- Plain `.txt` or `.jsonl` files
- Corpus must be line-based for traceability

### Output
- JSON: per-token activations
- CSV: neuron summary, top-K tokens
- PNG/SVG: heatmaps, activation maps
- HTML: interactive dashboards
- Folder: `/output/<run_name>/` for each analysis

---

## 🔗 Modularity Rules

- `/internal/models/`: model adapters
- `/internal/analysis/`: activation logic
- `/internal/metrics/`: sparsity, drift, etc.
- `/visuals/`: output images and plots
- `/cmd/nscope/`: CLI logic and command entrypoints

---

## 🧪 Priority Use Cases / Examples

- Find which tokens activate neuron 1574 in GPT-2 layer 5
- Visualize polysemantic neurons that respond to unrelated topics
- Generate heatmaps of neuron activation from a sample corpus
- Trace neuron reuse across two versions of the same model
- Build cluster maps of co-activated neurons across token spans

---

## 🤖 Prompt Hooks for AI Tools

> Implement a CLI command to run neuron-to-token reverse queries

> Write a Plotly visualization that shows neuron drift across two model versions

> Create a JSON schema for storing token span, layer, neuron ID, and activation value

> Add support for a new HuggingFace-compatible model in `/internal/models/llama.py`

> How should we detect polysemantic neurons from an activation matrix?

---

## ✅ Developer Tips

- Follow GoDoc-style or PEP257-style docstrings
- Organize output clearly for debugging
- Add examples to `/examples/` and results to `/output/`
- Use real tokens from common corpora: colors, professions, news headlines, etc.

---

This guide should be loaded as early context for all dev tools working on NeuronScope.
