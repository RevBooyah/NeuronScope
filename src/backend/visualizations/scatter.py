"""
Neuron Activation Scatter Plot Visualization

This module creates scatter plot visualizations of neuron activations
using dimensionality reduction techniques (PCA, t-SNE) for GPT-2 models.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Any, Optional, Tuple, Literal
from pathlib import Path

logger = logging.getLogger(__name__)

class ActivationScatter:
    """Creates scatter plot visualizations of neuron activations."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the scatter plot visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        
    def create_pca_scatter(self, activation_data: Dict[str, Any],
                          layer_index: int = 0,
                          max_neurons: int = 500,
                          save_path: Optional[str] = None) -> str:
        """
        Create a PCA scatter plot for neuron activations in a specific layer.
        
        Args:
            activation_data: Activation data dictionary
            layer_index: Index of the layer to visualize
            max_neurons: Maximum number of neurons to include (for performance)
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
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(activation_matrix)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create scatter plot
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                               c=np.mean(activation_matrix, axis=1), 
                               cmap='viridis', alpha=0.7, s=50)
            
            # Set labels
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
            ax.set_title(f'PCA of Neuron Activations - Layer {layer_index}\nPrompt: "{activation_data["prompt"]}"')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Mean Activation')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved PCA scatter plot to: {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Failed to create PCA scatter plot: {str(e)}")
            raise RuntimeError(f"PCA scatter plot creation failed: {str(e)}")
    
    def create_tsne_scatter(self, activation_data: Dict[str, Any],
                           layer_index: int = 0,
                           max_neurons: int = 300,
                           perplexity: float = 30.0,
                           save_path: Optional[str] = None) -> str:
        """
        Create a t-SNE scatter plot for neuron activations in a specific layer.
        
        Args:
            activation_data: Activation data dictionary
            layer_index: Index of the layer to visualize
            max_neurons: Maximum number of neurons to include (for performance)
            perplexity: t-SNE perplexity parameter
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
            
            # Limit neurons for performance (t-SNE is computationally expensive)
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
            
            # Apply t-SNE
            logger.info(f"Running t-SNE with {num_neurons} neurons, {num_tokens} tokens, perplexity={perplexity}")
            tsne = TSNE(n_components=2, perplexity=min(perplexity, num_neurons-1), 
                       random_state=42, max_iter=1000)
            tsne_result = tsne.fit_transform(activation_matrix)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create scatter plot
            scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                               c=np.mean(activation_matrix, axis=1), 
                               cmap='viridis', alpha=0.7, s=50)
            
            # Set labels
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f't-SNE of Neuron Activations - Layer {layer_index}\nPrompt: "{activation_data["prompt"]}"')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Mean Activation')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved t-SNE scatter plot to: {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Failed to create t-SNE scatter plot: {str(e)}")
            raise RuntimeError(f"t-SNE scatter plot creation failed: {str(e)}")
    
    def create_comparison_scatter(self, activation_data: Dict[str, Any],
                                layer_indices: List[int] = [0, 1, 2],
                                method: Literal['pca', 'tsne'] = 'pca',
                                max_neurons_per_layer: int = 200,
                                save_path: Optional[str] = None) -> str:
        """
        Create comparison scatter plots for multiple layers.
        
        Args:
            activation_data: Activation data dictionary
            layer_indices: List of layer indices to visualize
            method: Dimensionality reduction method ('pca' or 'tsne')
            max_neurons_per_layer: Maximum neurons per layer to include
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved image file
        """
        try:
            tokens = activation_data['tokens']
            num_layers = len(layer_indices)
            
            # Create subplot grid
            cols = min(3, num_layers)
            rows = (num_layers + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            
            if num_layers == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.reshape(rows, cols)
            
            # Process each layer
            for i, layer_idx in enumerate(layer_indices):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
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
                activation_matrix = np.zeros((num_neurons, len(tokens)))
                for j, neuron in enumerate(neurons):
                    activations = np.array(neuron['activations'])
                    activation_matrix[j, :] = activations
                
                # Apply dimensionality reduction
                if method == 'pca':
                    reducer = PCA(n_components=2)
                    result = reducer.fit_transform(activation_matrix)
                    explained_variance = reducer.explained_variance_ratio_
                    xlabel = f'PC1 ({explained_variance[0]:.1%})'
                    ylabel = f'PC2 ({explained_variance[1]:.1%})'
                else:  # tsne
                    reducer = TSNE(n_components=2, perplexity=min(30, num_neurons-1), 
                                 random_state=42, max_iter=1000)
                    result = reducer.fit_transform(activation_matrix)
                    xlabel = 't-SNE 1'
                    ylabel = 't-SNE 2'
                
                # Create scatter plot
                scatter = ax.scatter(result[:, 0], result[:, 1], 
                                   c=np.mean(activation_matrix, axis=1), 
                                   cmap='viridis', alpha=0.7, s=30)
                
                # Set labels
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'Layer {layer_idx} ({num_neurons} neurons)')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Mean Activation')
            
            # Hide empty subplots
            for i in range(num_layers, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)
            
            # Set main title
            method_name = method.upper()
            fig.suptitle(f'{method_name} Comparison Across Layers\nPrompt: "{activation_data["prompt"]}"', 
                        fontsize=14, y=0.98)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved comparison scatter plot to: {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Failed to create comparison scatter plot: {str(e)}")
            raise RuntimeError(f"Comparison scatter plot creation failed: {str(e)}")

def create_scatter_visualizer(figsize: Tuple[int, int] = (10, 8)) -> ActivationScatter:
    """Factory function to create an ActivationScatter instance."""
    return ActivationScatter(figsize) 