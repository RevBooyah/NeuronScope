# NeuronScope

**NeuronScope** is an open-source research and visualization platform designed for exploring neuron activations inside transformer models (initially GPT-2, with support for LLaMA planned later). It aims to provide interactive, intuitive visual insights into how transformer model neurons activate, cluster, drift, and respond to various inputs.

![NeuronScope Dashboard](images/neuronscope_1.png)

## 🧠 Visualization Examples

### Heatmap Visualization
![Neuron Activation Heatmap](images/heatmap-example.png)
*Neuron activations across tokens in Layer 0 of GPT-2 for "Hello world"*

### Multi-Layer Comparison
![Multi-Layer Heatmap](images/multi-layer-example.png)
*Neuron activations across multiple layers showing how patterns evolve*

### Scatter Plot Analysis
![PCA Scatter Plot](images/scatter-example.png)
*PCA projection of neuron activations showing clustering patterns*

### Statistical Summary
![Activation Summary](images/summary-example.png)
*Statistical analysis of activation patterns across all layers*

## Project Objectives
- Offer detailed insight into transformer neuron behavior and patterns.
- Visualize neuron activations clearly, interactively, and meaningfully.
- Support researchers with a user-friendly, interactive web-based interface.
- Provide supplementary CLI tools for batch processing and automated analysis.

## Key Features (MVP)
- **Neuron Activation Visualization**
  - Generalized neuron activation heatmaps.
  - Interactive scatter plots and dimensionality reductions (e.g., PCA, t-SNE).
  - Polysemantic neuron detection and exploration.
- **Neuron Clustering**
  - Cluster neurons by activation similarity and visualize clearly.
  - Allow interactive exploration of neuron groups and clusters.
- **Reverse Activation Queries**
  - Identify tokens, bi-grams, or n-grams that strongly activate specific neurons or neuron clusters.
  - Interactive querying through intuitive UI.
- **Neuron Drift Analysis**
  - Static and animated visuals tracking neuron activations across model fine-tuning checkpoints.
  - Clearly visualize neuron evolution and drift patterns over time.

## 🚀 Current Status

**Phase 2 Complete!** NeuronScope now has a fully functional backend and interactive frontend:

### ✅ **Implemented Features**
- **GPT-2 Model Integration**: Load and extract activations from GPT-2 models
- **Static Visualizations**: Heatmaps, scatter plots, and statistical summaries
- **React Frontend**: Modern, responsive dashboard with interactive controls
- **Data Integration**: Load and visualize real activation data
- **Multi-Layer Analysis**: Explore activations across all 12 GPT-2 layers

### 🔄 **In Development**
- **Clustering Algorithms**: K-Means clustering of neurons
- **Reverse Activation Queries**: Find tokens that activate specific neurons
- **Advanced Visualizations**: Polysemantic neuron detection
- **Backend API**: HTTP API for real-time data generation

## Technology Stack
- **Backend/Core:** Python (PyTorch, NumPy, Pandas, scikit-learn)
- **Frontend/Web UI:** React with interactive plotting libraries (Plotly, D3.js)
- **Visualization:** Initial static visualizations, incrementally enhanced by animations.
- **Optional Optimization:** C or Cython only if substantial (>5x) performance gains identified.

## Quick Start

See [SETUP.md](SETUP.md) for detailed installation and setup instructions.

```bash
# Clone and setup
git clone <repository-url>
cd nscope
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify backend setup
python scripts/setup_models.py

# Extract sample activations
python scripts/extract_activations.py

# Start the React frontend
cd src/frontend
npm install
npm start
```

The application will be available at `http://localhost:3000`

## Project Workflow
- Start with GPT-2 for initial development and quick prototyping. LLaMA once things are stabalized.
- Implement a web-first UI/UX with minimal dependencies.
- Incrementally introduce advanced visualizations, animations, and additional model support (LLaMA, Mistral, etc.).

## Contributions & Collaboration
Initially, NeuronScope will be built for local, single-developer use. Contributions, Dockerization, comprehensive documentation, and more robust testing will follow after core MVP stabilization.

---
