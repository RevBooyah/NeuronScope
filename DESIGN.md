# NeuronScope Design Overview

NeuronScope is structured around a modular separation between backend (data generation, analysis) and frontend (interactive visual display). This file outlines the architecture and major components.

---

## System Architecture

```
+----------------------------+
|        React UI           |
|  - Dashboard              |
|  - Heatmap Viewer         |
|  - Cluster Visualizer     |
|  - Reverse Query Tool     |
+------------+--------------+
             |
             | JSON Data (via local files or API)
             v
+------------+--------------+
|      Python Backend       |
|  - GPT-2 Model Loader     |
|  - Activation Extractor   |
|  - Clustering Engine      |
|  - Reverse Query Logic    |
+------------+--------------+
             |
             v
+----------------------------+
|      Output Files (JSON)   |
|  - activations/            |
|  - clusters/               |
|  - prompts/                |
+----------------------------+
```

---

## Technologies

### Backend
- Python 3.11+
- PyTorch (for GPT-2)
- NumPy / Pandas (for analysis)
- scikit-learn (clustering, PCA)
- Optionally: Cython for performance modules

### Frontend
- React + Vite or CRA
- Plotly or D3.js (for visualization)
- Tailwind (optional styling)

---

## Data Flow (Basic)
1. User enters a prompt (or selects a sample).
2. Python backend runs the prompt through GPT-2, extracting activations.
3. Backend saves structured JSON for:
   - Raw activations
   - Clustered neuron groupings
   - Top-triggering tokens (reverse queries)
4. React frontend loads and visualizes the data via interactive components.

---

## Visualization Design Principles
- Prioritize interpretability: clear axis labels, tooltips, scaling.
- Start with static visualizations; upgrade to animations after.
- Reuse visual styles (colors, shapes) between different views for familiarity.