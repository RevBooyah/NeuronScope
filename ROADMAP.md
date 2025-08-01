# NeuronScope Roadmap

This file outlines major milestones, features, and future plans for the NeuronScope project. Each phase builds on the previous, with room for exploration and iteration.

---

## ✅ Phase 1: MVP (Minimum Viable Platform)

**Goal:** Demonstrate neuron activation visualization and clustering for GPT-2 via a local web interface.

- [ ] Load GPT-2 and extract neuron activations for user and sample prompts.
- [ ] Output activation data in JSON format.
- [ ] Visualize activations as static heatmaps.
- [ ] Implement neuron clustering with KMeans.
- [ ] Visualize clusters as scatter plots.
- [ ] Support reverse-activation queries.
- [ ] Build a basic web UI with sample selector and visual embedding.

---

## 🔄 Phase 2: Visualization Expansion

**Goal:** Improve quality and depth of visualizations.

- [ ] Introduce dimensionality reduction (PCA, t-SNE, UMAP).
- [ ] Add drift comparison between checkpoints (static).
- [ ] Add simple animation of neuron drift over time.
- [ ] Evaluate visual UX and add user toggles, zoom, or filtering tools.
- [ ] Improve readability (legends, tooltips, color coding).

---

## 🚀 Phase 3: Model Support Expansion

**Goal:** Generalize backend for additional transformer models.

- [ ] Add support for LLaMA (via HuggingFace or GGUF).
- [ ] Standardize output format across models.
- [ ] Compare neuron behavior across model types.

---

## 📦 Phase 4: Packaging and Collaboration

**Goal:** Prepare for community use or collaboration.

- [ ] Add Docker support for backend/frontend.
- [ ] Add minimal CI/CD setup (e.g., test activation outputs or format).
- [ ] Improve documentation and developer onboarding instructions.
- [ ] Create example gallery of neuron phenomena.

---

## 🌱 Stretch Goals & Ideas

- [ ] Auto-discovery of interesting neurons.
- [ ] Train lightweight student models for interpretable activation behavior.
- [ ] Publish notebook/paper on NeuronScope visualizations.

---