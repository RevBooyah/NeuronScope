# NeuronScope Cursor Configuration & Usage Notes

This file contains guidance for using Cursor (AI IDE) effectively with this codebase.

---

## Cursor Usage Philosophy

NeuronScope is developed with clear, modular task definitions. Most work can be driven using small prompts directly inside Cursor. This project is designed to *not* rely on vague "vibe coding."

---

## Prompting Guidelines

- Use `TODO.md` to select precise development targets.
- Use `PROMPTS.md` for reusable prompt patterns that produce reliable output.

---

## Recommended Prompts in Cursor

- *React component prompt:*
  > Build a React component that displays a heatmap from this neuron activation JSON.

- *Data transformation prompt:*
  > Write a Python function to convert activation JSON into a 2D array for plotting.

- *Batch CLI prompt:*
  > Create a CLI tool that loads all prompts from `samples.json` and outputs their activations.

---

## Plugin Configuration

- Enable:
  - **Python tools**
  - **JSON viewer**
  - **React/TypeScript assistance**
- Optional:
  - LLMs for chat-backed autocomplete or inline fix suggestions.

---

## Tips

- Break tasks into one-feature files to improve LLM comprehension.
- Use comments or docstrings before prompting Cursor for modifications.
- Avoid mixing multiple unrelated logic blocks in one session—Cursor performs better when focused.

---

This file can be updated with effective prompt examples or plugin changes over time.