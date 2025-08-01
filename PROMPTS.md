# NeuronScope Prompt Templates for Cursor

These prompts are designed to be copy/paste-friendly inside Cursor when working on specific features.

---

## 📊 Visualization

**Static Heatmap**
> Create a React component that displays a heatmap of neuron activations from a JSON file. Each row should represent a neuron, each column a token. Use Plotly or a performant D3 alternative.

**Neuron Clustering Scatter Plot**
> Build a Python function that reduces high-dimensional neuron activations (e.g., 768-dim) using PCA or t-SNE and plots them with matplotlib or Plotly. Group points by cluster ID with color.

---

## 🧠 Model Interaction

**Activation Extraction**
> Write a Python function that takes a string prompt and returns the hidden states for all layers and all neurons in GPT-2.

**Top Token Activation Query**
> Write a function that returns the top N tokens that cause the highest activation for a given neuron across a list of prompts.

---

## 🔄 CLI Tooling

**Batch Activation Processor**
> Create a CLI tool in Python that takes a list of prompts, runs GPT-2 on each, and outputs their neuron activation JSON files.

**Cluster Automation**
> Write a CLI script that loads activation JSON files, performs clustering per layer using KMeans, and writes the results to `clusters/`.

---

## 📦 Frontend Glue

**Data Loader for Plotly Heatmap**
> Write a React hook that loads a local JSON file containing neuron activations and returns formatted data suitable for Plotly heatmap display.

**Dropdown for Prompt Selection**
> Create a React component that renders a dropdown from `samples.json` and updates the current prompt selection.

---

## 🧪 Test Data

**Synthetic Activation Generator**
> Create a Python script that generates random activation data in the same format as the real output, for UI testing.

---

These prompt patterns help keep Cursor focused and reusable. Add new ones as needed.