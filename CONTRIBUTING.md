# Contributing to NeuronScope

NeuronScope is a research tool and visualization platform designed to explore neuron activations inside transformer models. This document describes how development is organized and how to contribute effectively—even if you're the primary or only developer.

## Development Workflow

1. **Tasks should be small and focused.**  
   Each development task should ideally be a single feature, function, or visualization. Use TODO.md as your task backlog.

2. **Use Cursor for assistance.**  
   Cursor is expected to help implement tasks. Most items in TODO.md are written to be AI-prompt-friendly. You can use PROMPTS.md for prompt templates that work well.

3. **Branching Strategy (if using Git branches)**  
   - `main` → stable, working version.
   - `dev/*` → feature-specific branches for new functionality (e.g., `dev/activation-heatmap`).

4. **Commit Style**
   - Use concise, imperative commit messages.
     ```
     Add scatter plot for neuron clustering
     Fix bug in reverse activation query JSON output
     ```

5. **PRs or Merges**
   - PRs are optional unless collaborating.
   - If solo, treat PRs as checkpoints (optional).

## Cursor Prompt Format (Examples)

- *For UI work:*
  > Build a React component that displays a heatmap from a JSON file of neuron activations.

- *For backend logic:*
  > Write a Python function that extracts the hidden states of all neurons from GPT-2 for a given prompt.

---

Happy building!