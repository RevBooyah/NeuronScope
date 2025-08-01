"""
Visualization module for NeuronScope.

This module contains visualization tools for neuron activations,
clustering results, and other analysis outputs.
"""

from .heatmap import create_heatmap_visualizer, ActivationHeatmap
from .scatter import create_scatter_visualizer, ActivationScatter

__all__ = [
    'create_heatmap_visualizer', 
    'ActivationHeatmap',
    'create_scatter_visualizer',
    'ActivationScatter'
] 