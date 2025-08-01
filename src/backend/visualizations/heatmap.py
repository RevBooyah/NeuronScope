"""
Neuron Activation Heatmap Visualization

This module creates heatmap visualizations of neuron activations
across layers and tokens for GPT-2 models.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ActivationHeatmap:
    """Creates heatmap visualizations of neuron activations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the heatmap visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        
    def create_layer_heatmap(self, activation_data: Dict[str, Any], 
                           layer_index: int = 0,
                           max_neurons: int = 100,
                           save_path: Optional[str] = None) -> str:
        """
        Create a heatmap for a specific layer showing neuron activations across tokens.
        
        Args:
            activation_data: Activation data dictionary
            layer_index: Index of the layer to visualize
            max_neurons: Maximum number of neurons to show (for performance)
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved image file
        """
        try:
            # Extract layer data
            if layer_index >= len(activation_data['layers']):
                raise ValueError(f"Layer {layer_index} not found. Available layers: 0-{len(activation_data['layers'])-1}")
            
            layer_data = activation_data['layers'][layer_index]
            tokens = activation_data['tokens']
            
            # Extract activation matrix
            neurons = layer_data['neurons']
            num_neurons = len(neurons)
            num_tokens = len(tokens)
            
            # Limit neurons for performance
            if num_neurons > max_neurons:
                # Sample neurons evenly
                step = num_neurons // max_neurons
                neuron_indices = list(range(0, num_neurons, step))[:max_neurons]
                neurons = [neurons[i] for i in neuron_indices]
                num_neurons = len(neurons)
                logger.info(f"Sampled {num_neurons} neurons from {len(layer_data['neurons'])} total")
            
            # Create activation matrix
            activation_matrix = np.zeros((num_neurons, num_tokens))
            for i, neuron in enumerate(neurons):
                activations = np.array(neuron['activations'])
                activation_matrix[i, :] = activations
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create heatmap
            im = ax.imshow(activation_matrix, cmap='RdBu_r', aspect='auto')
            
            # Set labels
            ax.set_xlabel('Tokens')
            ax.set_ylabel('Neurons')
            ax.set_title(f'Neuron Activations - Layer {layer_index}\nPrompt: "{activation_data["prompt"]}"')
            
            # Set tick labels
            ax.set_xticks(range(num_tokens))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticks(range(0, num_neurons, max(1, num_neurons // 10)))
            ax.set_yticklabels([f'N{i}' for i in range(0, num_neurons, max(1, num_neurons // 10))])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Activation Value')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved heatmap to: {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Failed to create layer heatmap: {str(e)}")
            raise RuntimeError(f"Heatmap creation failed: {str(e)}")
    
    def create_multi_layer_heatmap(self, activation_data: Dict[str, Any],
                                 layers_to_show: Optional[List[int]] = None,
                                 max_neurons_per_layer: int = 50,
                                 save_path: Optional[str] = None) -> str:
        """
        Create a multi-layer heatmap showing activations across multiple layers.
        
        Args:
            activation_data: Activation data dictionary
            layers_to_show: List of layer indices to show. If None, shows all layers.
            max_neurons_per_layer: Maximum neurons per layer to show
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved image file
        """
        try:
            tokens = activation_data['tokens']
            num_tokens = len(tokens)
            
            # Determine which layers to show
            if layers_to_show is None:
                layers_to_show = list(range(len(activation_data['layers'])))
            
            num_layers = len(layers_to_show)
            
            # Create subplot grid
            fig, axes = plt.subplots(num_layers, 1, figsize=(12, 3 * num_layers))
            if num_layers == 1:
                axes = [axes]
            
            # Process each layer
            for i, layer_idx in enumerate(layers_to_show):
                layer_data = activation_data['layers'][layer_idx]
                neurons = layer_data['neurons']
                num_neurons = len(neurons)
                
                # Sample neurons if needed
                if num_neurons > max_neurons_per_layer:
                    step = num_neurons // max_neurons_per_layer
                    neuron_indices = list(range(0, num_neurons, step))[:max_neurons_per_layer]
                    neurons = [neurons[j] for j in neuron_indices]
                    num_neurons = len(neurons)
                
                # Create activation matrix
                activation_matrix = np.zeros((num_neurons, num_tokens))
                for j, neuron in enumerate(neurons):
                    activations = np.array(neuron['activations'])
                    activation_matrix[j, :] = activations
                
                # Create heatmap for this layer
                im = axes[i].imshow(activation_matrix, cmap='RdBu_r', aspect='auto')
                
                # Set labels
                axes[i].set_title(f'Layer {layer_idx} ({num_neurons} neurons)')
                if i == num_layers - 1:  # Only show x-labels on bottom subplot
                    axes[i].set_xlabel('Tokens')
                    axes[i].set_xticks(range(num_tokens))
                    axes[i].set_xticklabels(tokens, rotation=45, ha='right')
                else:
                    axes[i].set_xticks([])
                
                axes[i].set_ylabel('Neurons')
                axes[i].set_yticks(range(0, num_neurons, max(1, num_neurons // 5)))
                axes[i].set_yticklabels([f'N{j}' for j in range(0, num_neurons, max(1, num_neurons // 5))])
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[i])
                cbar.set_label('Activation')
            
            # Set main title
            fig.suptitle(f'Multi-Layer Neuron Activations\nPrompt: "{activation_data["prompt"]}"', 
                        fontsize=14, y=0.98)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved multi-layer heatmap to: {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Failed to create multi-layer heatmap: {str(e)}")
            raise RuntimeError(f"Multi-layer heatmap creation failed: {str(e)}")
    
    def create_activation_summary(self, activation_data: Dict[str, Any],
                                save_path: Optional[str] = None) -> str:
        """
        Create a summary visualization showing activation statistics across layers.
        
        Args:
            activation_data: Activation data dictionary
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved image file
        """
        try:
            layers = activation_data['layers']
            num_layers = len(layers)
            
            # Calculate statistics for each layer
            layer_stats = []
            for layer in layers:
                activations = []
                for neuron in layer['neurons']:
                    activations.extend(neuron['activations'])
                
                activations = np.array(activations)
                layer_stats.append({
                    'mean': np.mean(activations),
                    'std': np.std(activations),
                    'max': np.max(activations),
                    'min': np.min(activations),
                    'abs_mean': np.mean(np.abs(activations))
                })
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Mean activation per layer
            means = [stats['mean'] for stats in layer_stats]
            ax1.plot(range(num_layers), means, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Mean Activation')
            ax1.set_title('Mean Activation per Layer')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Standard deviation per layer
            stds = [stats['std'] for stats in layer_stats]
            ax2.plot(range(num_layers), stds, 'r-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Standard Deviation')
            ax2.set_title('Activation Standard Deviation per Layer')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Activation range per layer
            maxs = [stats['max'] for stats in layer_stats]
            mins = [stats['min'] for stats in layer_stats]
            ax3.fill_between(range(num_layers), mins, maxs, alpha=0.3, color='green')
            ax3.plot(range(num_layers), maxs, 'g-', linewidth=2, label='Max')
            ax3.plot(range(num_layers), mins, 'g-', linewidth=2, label='Min')
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Activation Value')
            ax3.set_title('Activation Range per Layer')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Absolute mean activation per layer
            abs_means = [stats['abs_mean'] for stats in layer_stats]
            ax4.bar(range(num_layers), abs_means, color='purple', alpha=0.7)
            ax4.set_xlabel('Layer')
            ax4.set_ylabel('Absolute Mean Activation')
            ax4.set_title('Absolute Mean Activation per Layer')
            ax4.grid(True, alpha=0.3)
            
            # Set main title
            fig.suptitle(f'Activation Summary Statistics\nPrompt: "{activation_data["prompt"]}"', 
                        fontsize=14, y=0.98)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved activation summary to: {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Failed to create activation summary: {str(e)}")
            raise RuntimeError(f"Activation summary creation failed: {str(e)}")

def create_heatmap_visualizer(figsize: Tuple[int, int] = (12, 8)) -> ActivationHeatmap:
    """Factory function to create an ActivationHeatmap instance."""
    return ActivationHeatmap(figsize) 