# NeuronScope Visualization Guidelines

These guidelines ensure that all visual outputs in NeuronScope are consistent, readable, and informative.

---

## General Principles

- **Clarity over aesthetics:** Prioritize clean visuals with understandable legends and axis labels.
- **Consistency across views:** Use the same colors, fonts, scales, and styles where possible to reduce cognitive load.
- **Scalability:** Design visuals that work with a small number of neurons or a full transformer layer.
- **Accessibility:** Use color palettes that are colorblind-friendly and distinguishable in grayscale when possible.

---

## Heatmaps

- Axes:
  - X-axis → Token index or token string
  - Y-axis → Neuron index (grouped by layer if needed)
- Use diverging color scales (e.g., red–white–blue) centered at zero for signed activations.
- Add tooltips showing token, neuron index, and activation value.

---

## Scatter Plots (Clustering)

- Use PCA or t-SNE for dimensionality reduction.
- Color-code by cluster ID or layer.
- Tooltip should include:
  - Neuron index
  - Cluster ID
  - Layer
- Use point size sparingly (if used, encode by activation variance or salience).

---

## Reverse Activation Query Visuals

- Display top N activating tokens as a bar chart or table.
- Include activation score in the tooltip or label.
- Allow toggling between neuron and cluster-based queries.

---

## Drift Visualizations

- Static drift:
  - Use side-by-side or overlaid scatter plots to compare checkpoints.
- Animated drift:
  - Use fading or morphing transitions for neurons that move over time.
  - Annotate neurons that move significantly between checkpoints.

---

## Suggested Tools & Libraries

- **Plotly.js** for interactivity in React.
- **Matplotlib or Seaborn** for quick testing in Python.
- **D3.js** (only if needed for advanced custom visuals).

---

These guidelines will evolve as user feedback and new features emerge.