"""
Weight Analysis Module for Pruning Analysis

This module provides functionality to analyze model weights for pruning purposes,
including weight magnitude analysis, sparsity detection, and pruning candidate identification.
"""

import torch
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WeightStats:
    """Statistics for a layer's weights."""
    layer_name: str
    layer_index: int
    total_parameters: int
    non_zero_parameters: int
    sparsity: float
    mean_magnitude: float
    std_magnitude: float
    min_magnitude: float
    max_magnitude: float
    l1_norm: float
    l2_norm: float

@dataclass
class NeuronWeightInfo:
    """Information about a specific neuron's weights."""
    layer_index: int
    neuron_index: int
    weight_magnitude: float
    weight_rank: int  # Rank within layer (1 = largest magnitude)
    is_pruning_candidate: bool
    pruning_score: float

class WeightAnalyzer:
    """Analyzes model weights for pruning analysis."""
    
    def __init__(self, model):
        """Initialize with a PyTorch model."""
        self.model = model
        self.device = next(model.parameters()).device
        
    def extract_weight_stats(self) -> Dict[str, WeightStats]:
        """Extract comprehensive weight statistics for all layers."""
        stats = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.detach().cpu().numpy()
                
                # Calculate statistics
                total_params = weight.size
                non_zero_params = np.count_nonzero(weight)
                sparsity = 1.0 - (non_zero_params / total_params)
                
                magnitudes = np.abs(weight)
                mean_mag = np.mean(magnitudes)
                std_mag = np.std(magnitudes)
                min_mag = np.min(magnitudes)
                max_mag = np.max(magnitudes)
                
                # Calculate norms
                l1_norm = np.sum(magnitudes)
                l2_norm = np.sqrt(np.sum(magnitudes ** 2))
                
                stats[name] = WeightStats(
                    layer_name=name,
                    layer_index=self._extract_layer_index(name),
                    total_parameters=total_params,
                    non_zero_parameters=non_zero_params,
                    sparsity=sparsity,
                    mean_magnitude=mean_mag,
                    std_magnitude=std_mag,
                    min_magnitude=min_mag,
                    max_magnitude=max_mag,
                    l1_norm=l1_norm,
                    l2_norm=l2_norm
                )
        
        return stats
    
    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index from layer name."""
        try:
            # Handle different naming conventions
            if 'transformer.h.' in layer_name:
                parts = layer_name.split('.')
                for i, part in enumerate(parts):
                    if part == 'h' and i + 1 < len(parts):
                        return int(parts[i + 1])
            elif 'layer.' in layer_name:
                parts = layer_name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts):
                        return int(parts[i + 1])
        except (ValueError, IndexError):
            pass
        return -1
    
    def get_weight_magnitudes(self, layer_name: str) -> np.ndarray:
        """Get weight magnitudes for a specific layer."""
        for name, module in self.model.named_modules():
            if name == layer_name and hasattr(module, 'weight'):
                return np.abs(module.weight.detach().cpu().numpy())
        return np.array([])
    
    def identify_pruning_candidates(self, 
                                  threshold_percentile: float = 10.0,
                                  min_sparsity: float = 0.1) -> List[NeuronWeightInfo]:
        """Identify neurons that are good candidates for pruning."""
        candidates = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.detach().cpu().numpy()
                layer_index = self._extract_layer_index(name)
                
                if layer_index >= 0:
                    # Analyze each neuron in the layer
                    if len(weight.shape) >= 2:
                        # For linear layers, analyze output neurons
                        for neuron_idx in range(weight.shape[0]):
                            neuron_weights = weight[neuron_idx]
                            magnitude = np.mean(np.abs(neuron_weights))
                            
                            # Calculate percentile rank
                            all_magnitudes = np.mean(np.abs(weight), axis=1)
                            rank = np.percentile(all_magnitudes, 
                                               (magnitude / np.max(all_magnitudes)) * 100)
                            
                            # Determine if this is a pruning candidate
                            is_candidate = (rank <= threshold_percentile or 
                                          magnitude <= np.percentile(all_magnitudes, threshold_percentile))
                            
                            # Calculate pruning score (lower = more likely to prune)
                            pruning_score = 1.0 - (magnitude / np.max(all_magnitudes))
                            
                            candidates.append(NeuronWeightInfo(
                                layer_index=layer_index,
                                neuron_index=neuron_idx,
                                weight_magnitude=magnitude,
                                weight_rank=int(rank),
                                is_pruning_candidate=is_candidate,
                                pruning_score=pruning_score
                            ))
        
        return candidates
    
    def get_sparsity_analysis(self) -> Dict[str, Any]:
        """Get comprehensive sparsity analysis."""
        stats = self.extract_weight_stats()
        
        # Calculate overall sparsity
        total_params = sum(s.total_parameters for s in stats.values())
        total_non_zero = sum(s.non_zero_parameters for s in stats.values())
        overall_sparsity = 1.0 - (total_non_zero / total_params)
        
        # Layer-wise sparsity
        layer_sparsity = {
            s.layer_name: {
                'sparsity': float(s.sparsity),
                'total_params': int(s.total_parameters),
                'non_zero_params': int(s.non_zero_parameters)
            }
            for s in stats.values()
        }
        
        return {
            'overall_sparsity': float(overall_sparsity),
            'total_parameters': int(total_params),
            'non_zero_parameters': int(total_non_zero),
            'layer_sparsity': layer_sparsity,
            'layer_stats': [
                {
                    'layer_name': s.layer_name,
                    'layer_index': int(s.layer_index),
                    'total_parameters': int(s.total_parameters),
                    'non_zero_parameters': int(s.non_zero_parameters),
                    'sparsity': float(s.sparsity),
                    'mean_magnitude': float(s.mean_magnitude),
                    'std_magnitude': float(s.std_magnitude),
                    'min_magnitude': float(s.min_magnitude),
                    'max_magnitude': float(s.max_magnitude),
                    'l1_norm': float(s.l1_norm),
                    'l2_norm': float(s.l2_norm)
                }
                for s in stats.values()
            ]
        }
    
    def export_weight_analysis(self, output_path: str) -> None:
        """Export weight analysis results to JSON."""
        analysis = {
            'sparsity_analysis': self.get_sparsity_analysis(),
            'pruning_candidates': [
                {
                    'layer_index': c.layer_index,
                    'neuron_index': c.neuron_index,
                    'weight_magnitude': float(c.weight_magnitude),
                    'weight_rank': c.weight_rank,
                    'is_pruning_candidate': c.is_pruning_candidate,
                    'pruning_score': float(c.pruning_score)
                }
                for c in self.identify_pruning_candidates()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Weight analysis exported to {output_path}") 