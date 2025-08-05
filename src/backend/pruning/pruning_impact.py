"""
Pruning Impact Analysis Module

This module provides functionality to simulate pruning effects and analyze their impact
on model performance, including activation changes and neuron importance analysis.
"""

import torch
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)

@dataclass
class PruningImpact:
    """Results of pruning impact analysis."""
    original_activations: np.ndarray
    pruned_activations: np.ndarray
    activation_change: np.ndarray
    mean_change: float
    max_change: float
    affected_neurons: List[int]
    impact_score: float

@dataclass
class NeuronImportance:
    """Importance metrics for a neuron."""
    layer_index: int
    neuron_index: int
    activation_magnitude: float
    activation_variance: float
    impact_score: float
    is_critical: bool

class PruningImpactAnalyzer:
    """Analyzes the impact of pruning on model behavior."""
    
    def __init__(self, model, tokenizer):
        """Initialize with model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def simulate_neuron_pruning(self, 
                              layer_index: int, 
                              neuron_indices: List[int],
                              input_text: str) -> PruningImpact:
        """Simulate pruning specific neurons and measure impact."""
        
        # Get original activations
        original_activations = self._extract_activations(input_text)
        
        # Create a copy of the model for pruning simulation
        pruned_model = copy.deepcopy(self.model)
        
        # Apply pruning by zeroing out specific neurons
        self._apply_neuron_pruning(pruned_model, layer_index, neuron_indices)
        
        # Get activations after pruning
        pruned_activations = self._extract_activations_with_model(input_text, pruned_model)
        
        # Calculate impact metrics
        activation_change = np.abs(pruned_activations - original_activations)
        mean_change = np.mean(activation_change)
        max_change = np.max(activation_change)
        
        # Calculate impact score (normalized by original activation magnitude)
        impact_score = np.mean(activation_change / (np.abs(original_activations) + 1e-8))
        
        return PruningImpact(
            original_activations=original_activations,
            pruned_activations=pruned_activations,
            activation_change=activation_change,
            mean_change=mean_change,
            max_change=max_change,
            affected_neurons=neuron_indices,
            impact_score=impact_score
        )
    
    def _extract_activations(self, input_text: str) -> np.ndarray:
        """Extract activations for the given input text."""
        # This is a simplified version - in practice, you'd use the existing activation extractor
        tokens = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy())
        
        # Register hooks for all transformer layers
        hooks = []
        for name, module in self.model.named_modules():
            if 'mlp.c_fc' in name:  # Output of MLP layers
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            self.model(tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return np.concatenate(activations, axis=0) if activations else np.array([])
    
    def _extract_activations_with_model(self, input_text: str, model) -> np.ndarray:
        """Extract activations using a specific model."""
        tokens = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy())
        
        # Register hooks for all transformer layers
        hooks = []
        for name, module in model.named_modules():
            if 'mlp.c_fc' in name:  # Output of MLP layers
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            model(tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return np.concatenate(activations, axis=0) if activations else np.array([])
    
    def _apply_neuron_pruning(self, model, layer_index: int, neuron_indices: List[int]):
        """Apply pruning by zeroing out specific neurons in a layer."""
        for name, module in model.named_modules():
            if f'transformer.h.{layer_index}.mlp.c_fc' in name:
                # Zero out the specified neurons
                with torch.no_grad():
                    module.weight[neuron_indices, :] = 0.0
                    if module.bias is not None:
                        module.bias[neuron_indices] = 0.0
                break
    
    def analyze_neuron_importance(self, 
                                input_texts: List[str],
                                layer_index: int) -> List[NeuronImportance]:
        """Analyze the importance of neurons in a specific layer."""
        importance_scores = []
        
        # Get model info to determine number of neurons
        for name, module in self.model.named_modules():
            if f'transformer.h.{layer_index}.mlp.c_fc' in name:
                num_neurons = module.weight.shape[0]
                break
        else:
            return []
        
        # Analyze each neuron
        for neuron_idx in range(num_neurons):
            activation_magnitudes = []
            activation_variances = []
            impact_scores = []
            
            # Test with multiple input texts
            for text in input_texts:
                # Simulate pruning this single neuron
                impact = self.simulate_neuron_pruning(layer_index, [neuron_idx], text)
                
                activation_magnitudes.append(np.mean(np.abs(impact.original_activations)))
                activation_variances.append(np.var(impact.original_activations))
                impact_scores.append(impact.impact_score)
            
            # Calculate average metrics
            avg_magnitude = np.mean(activation_magnitudes)
            avg_variance = np.mean(activation_variances)
            avg_impact = np.mean(impact_scores)
            
            # Determine if neuron is critical (high impact when pruned)
            is_critical = avg_impact > np.percentile(impact_scores, 75)
            
            importance_scores.append(NeuronImportance(
                layer_index=layer_index,
                neuron_index=neuron_idx,
                activation_magnitude=avg_magnitude,
                activation_variance=avg_variance,
                impact_score=avg_impact,
                is_critical=is_critical
            ))
        
        return importance_scores
    
    def batch_pruning_analysis(self, 
                             input_texts: List[str],
                             pruning_candidates: List[Dict],
                             batch_size: int = 10) -> Dict[str, Any]:
        """Perform batch analysis of pruning candidates."""
        results = {
            'individual_impacts': [],
            'cumulative_impact': None,
            'recommendations': []
        }
        
        # Analyze individual neurons
        for candidate in pruning_candidates:
            layer_idx = candidate['layer_index']
            neuron_idx = candidate['neuron_index']
            
            # Test with a subset of input texts for efficiency
            test_texts = input_texts[:min(5, len(input_texts))]
            impact = self.simulate_neuron_pruning(layer_idx, [neuron_idx], test_texts[0])
            
            results['individual_impacts'].append({
                'layer_index': layer_idx,
                'neuron_index': neuron_idx,
                'impact_score': float(impact.impact_score),
                'mean_change': float(impact.mean_change),
                'max_change': float(impact.max_change),
                'safe_to_prune': impact.impact_score < 0.1  # Threshold for safe pruning
            })
        
        # Analyze cumulative impact of pruning multiple neurons
        if len(pruning_candidates) > 1:
            all_neuron_indices = [(c['layer_index'], c['neuron_index']) 
                                for c in pruning_candidates[:batch_size]]
            
            # Group by layer for batch pruning
            layer_groups = {}
            for layer_idx, neuron_idx in all_neuron_indices:
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = []
                layer_groups[layer_idx].append(neuron_idx)
            
            # Test cumulative impact
            test_text = input_texts[0] if input_texts else "Hello world"
            cumulative_impact = self.simulate_neuron_pruning(
                list(layer_groups.keys())[0], 
                list(layer_groups.values())[0], 
                test_text
            )
            
            results['cumulative_impact'] = {
                'impact_score': float(cumulative_impact.impact_score),
                'mean_change': float(cumulative_impact.mean_change),
                'max_change': float(cumulative_impact.max_change)
            }
        
        # Generate recommendations
        safe_candidates = [r for r in results['individual_impacts'] if r['safe_to_prune']]
        risky_candidates = [r for r in results['individual_impacts'] if not r['safe_to_prune']]
        
        results['recommendations'] = {
            'safe_to_prune': len(safe_candidates),
            'risky_to_prune': len(risky_candidates),
            'suggested_batch_size': min(10, len(safe_candidates)),
            'high_impact_neurons': [r for r in risky_candidates if r['impact_score'] > 0.5]
        }
        
        return results
    
    def export_pruning_analysis(self, 
                              analysis_results: Dict[str, Any], 
                              output_path: str) -> None:
        """Export pruning analysis results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Pruning analysis exported to {output_path}") 