# NeuronScope TODO List

Each task below should be specific enough to complete clearly in 1-2 prompts.

## Repo & Initial Setup
- [x] Initialize GitHub repository structure with directories (`src`, `data`, `scripts`, `ui`).
- [x] Set up Python virtual environment and install base packages (PyTorch, NumPy, Pandas, scikit-learn).

## Data & Model Setup
- [x] Write a CLI script to download and set up GPT-2 pretrained model locally.
- [x] Write basic data processing scripts to tokenize sample prompts and run model inference.
- [x] Define and document a small set of "common prompts" for quick visual testing (e.g., "Hello world", "Translate this sentence", "Summarize this paragraph").

## Neuron Activation Extraction
- [x] Write a function/script that extracts neuron activations for a given input token sequence from GPT-2.
- [x] Output neuron activation data in structured JSON/CSV format for easy visualization.

## Visualization Prototypes
- [x] Create a heatmap visualization script (static version) showing generalized neuron activations.
- [x] Prototype a scatter plot visualization using PCA or t-SNE to visualize neuron activations.
- [ ] Prototype a polysemantic neuron detection visualization.

## Web UI Setup
- [x] Initialize a basic React app project setup (use Create React App or Vite).
- [x] Set up initial React pages/routes: Dashboard/Home, Neuron Activations, Neuron Clustering, Reverse Queries.
- [x] Add Plotly or similar library integration to the React app.

## Model Information System
- [x] Create comprehensive model information service with architecture details, activation equations, training information.
- [x] Implement React modal component with tabbed interface for displaying model information.
- [x] Add model info button to dashboard header with responsive design.
- [x] Include detailed information about GPT-2 architecture, mathematical equations, training details, and usage guidelines.

## Visualization Integration (Web UI)
- [x] Connect backend (Python) neuron activation output data to the React frontend via a simple JSON data exchange.
- [x] Implement the generalized neuron activation heatmap visualization in the web UI.
- [x] Implement scatter plot visualization for neuron clustering in the web UI.

## Reverse Neuron Activation Queries
- [ ] Create backend script/function to query neuron data for top tokens/n-grams activating specific neurons.
- [ ] Implement initial simple query UI on React frontend to demonstrate reverse activation querying.

## Neuron Clustering
- [ ] Implement clustering of neurons (K-Means initially) in Python backend.
- [ ] Export clustering results (cluster assignments, cluster centers) in structured format for UI consumption.
- [ ] Visualize neuron clusters interactively in web UI (scatter plot, color-coded clusters).

## Drift Analysis Preparation
- [ ] Write a script to export neuron activations from multiple GPT-2 checkpoints (e.g., after fine-tuning epochs).
- [ ] Prototype static visualization comparing neuron activations across two checkpoints (heatmap or scatter plot).

## Animation Experiments
- [ ] Prototype a basic animation showing neuron activations changing over two or more model checkpoints.
- [ ] Evaluate effectiveness/usability of animations vs. static visualizations.

## CLI Tools for Automation & Testing
- [ ] Write CLI script to batch-process multiple prompts and store neuron activation outputs.
- [ ] Write CLI script to automate clustering analysis for multiple input datasets/prompts.

## Optimization Exploration (Optional)
- [ ] Benchmark Python implementation for neuron extraction/visualization performance.
- [ ] Explore potential C/Cython implementation if benchmarks suggest at least a 10x improvement in speed.

## Documentation & Examples (Minimal)
- [ ] Write a brief initial tutorial or quick-start guide demonstrating basic usage and visualizations.
- [ ] Add inline code comments clearly documenting the functions and data formats used.

---
