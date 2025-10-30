"""
Cross-model neuron comparison algorithms for NeuronScope.

This module computes similarity metrics between neurons across two models
using shared prompts. Initial metric: cosine similarity over concatenated
activation vectors per neuron across prompts/tokens. Layer alignment uses
min(#layers) heuristic by default.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from ..models.gpt2_loader import MultiModelLoader
from ..activations.extractor import ActivationExtractor
from ..utils.cache import cache_manager

logger = logging.getLogger(__name__)


class CrossModelComparator:
    def __init__(self, model_loader: MultiModelLoader):
        self.model_loader = model_loader

    def _get_extractor(self, model_name: str) -> ActivationExtractor:
        model, tokenizer = self.model_loader.load_model(model_name)
        return ActivationExtractor(model, tokenizer)

    def _collect_layer_activations(
        self,
        extractor: ActivationExtractor,
        prompts: List[str]
    ) -> List[np.ndarray]:
        """
        Returns list of length L where each item is an array of shape
        (num_neurons_in_layer, total_time_steps) for that layer, concatenated across prompts.
        """
        # Cache per (model, prompt) activation to avoid recomputation across runs
        per_prompt = []
        for p in prompts:
            cache_params = {"model": extractor.model.__class__.__name__, "prompt": p}
            cached = cache_manager.get("compare_prompt_activation", cache_params)
            if cached is not None:
                per_prompt.append(cached)
            else:
                act = extractor.extract_activations(p)
                per_prompt.append(act)
                cache_manager.set("compare_prompt_activation", cache_params, act)

        # Determine number of layers from first result
        num_layers = len(per_prompt[0]["layers"]) if per_prompt else 0
        layer_concat: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            per_neuron_series: List[np.ndarray] = []
            num_neurons = len(per_prompt[0]["layers"][layer_idx]["neurons"]) if num_layers > 0 else 0

            # For each neuron, concatenate activations across prompts along time axis
            for neuron_idx in range(num_neurons):
                series_list: List[np.ndarray] = []
                for act in per_prompt:
                    series = np.asarray(act["layers"][layer_idx]["neurons"][neuron_idx]["activations"], dtype=np.float32)
                    series_list.append(series)
                per_neuron_series.append(np.concatenate(series_list, axis=0))

            # Stack into (num_neurons, T_total)
            if per_neuron_series:
                layer_concat.append(np.stack(per_neuron_series, axis=0))
            else:
                layer_concat.append(np.zeros((0, 0), dtype=np.float32))

        return layer_concat

    @staticmethod
    def _cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between rows of A and rows of B.
        A: (N, D), B: (M, D) -> (N, M)
        """
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)

        # Normalize rows
        def normalize(X: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            return X / norms

        A_n = normalize(A)
        B_n = normalize(B)
        return A_n @ B_n.T

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        prompts: List[str],
        layer_alignment: str = "min_layers",
        metric: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Compare neurons across two models using shared prompts.

        Returns JSON with per-layer similarity matrices and summary stats.
        """
        extractor_a = self._get_extractor(model_a)
        extractor_b = self._get_extractor(model_b)

        layers_a = self._collect_layer_activations(extractor_a, prompts)
        layers_b = self._collect_layer_activations(extractor_b, prompts)

        if layer_alignment == "min_layers":
            num_layers = min(len(layers_a), len(layers_b))
        else:
            num_layers = min(len(layers_a), len(layers_b))

        results: List[Dict[str, Any]] = []
        for layer_idx in range(num_layers):
            A = layers_a[layer_idx]
            B = layers_b[layer_idx]

            if metric == "cosine":
                sim = self._cosine_similarity_matrix(A, B)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            layer_result = {
                "layer_index": layer_idx,
                "model_a_neurons": int(A.shape[0]),
                "model_b_neurons": int(B.shape[0]),
                "similarity": sim.tolist(),
                "summary": {
                    "mean_similarity": float(np.mean(sim)) if sim.size else 0.0,
                    "max_similarity": float(np.max(sim)) if sim.size else 0.0
                }
            }
            results.append(layer_result)

        return {
            "models": {"a": model_a, "b": model_b},
            "prompts": prompts,
            "metric": metric,
            "layer_alignment": layer_alignment,
            "layers": results
        }


