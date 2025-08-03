"""
Reverse activation queries for NeuronScope

This module implements functionality to find tokens, n-grams, or sequences
that strongly activate specific neurons or neuron clusters.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json

from models.gpt2_loader import GPT2ModelLoader
from activations.extractor import ActivationExtractor
from utils.data_io import DataIO

logger = logging.getLogger(__name__)

class ReverseQueryEngine:
    """Engine for performing reverse activation queries."""
    
    def __init__(self, model_loader: Optional[GPT2ModelLoader] = None, data_io: Optional[DataIO] = None):
        """
        Initialize the reverse query engine.
        
        Args:
            model_loader: GPT-2 model loader instance
            data_io: Data I/O handler instance
        """
        self.model_loader = model_loader or GPT2ModelLoader()
        self.data_io = data_io or DataIO()
        self.extractor = None  # Will be set when model is loaded
        
        # Common tokens for testing (can be expanded)
        self.test_tokens = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "among", "within", "without", "against", "toward",
            "hello", "world", "good", "bad", "big", "small", "new", "old", "high", "low",
            "first", "last", "next", "previous", "current", "future", "past", "present",
            "time", "space", "place", "thing", "person", "people", "man", "woman", "child",
            "house", "car", "book", "computer", "phone", "food", "water", "air", "fire",
            "earth", "sun", "moon", "star", "tree", "flower", "animal", "bird", "fish",
            "dog", "cat", "horse", "cow", "sheep", "pig", "chicken", "duck", "goose"
        ]
    
    def query_neuron_activations(self, neuron_index: int, layer_index: int, 
                                top_k: int = 10) -> Dict[str, Any]:
        """
        Find tokens that most strongly activate a specific neuron.
        
        Args:
            neuron_index: Index of the neuron to query
            layer_index: Layer index of the neuron
            top_k: Number of top activating tokens to return
            
        Returns:
            Dictionary with query results
        """
        logger.info(f"Querying neuron {neuron_index} in layer {layer_index}")
        
        # Load model and create extractor if not already done
        if self.extractor is None:
            model, tokenizer = self.model_loader.load_model('gpt2')
            self.extractor = ActivationExtractor(model, tokenizer)
        
        # Test each token and collect activations
        token_activations = []
        
        for token in self.test_tokens:
            try:
                # Extract activations for this token
                activation_data = self.extractor.extract_activations(token)
                
                # Get activation for the specific neuron
                layer_data = activation_data["layers"][layer_index]
                neuron_data = layer_data["neurons"][neuron_index]
                activation = neuron_data["activations"][0]  # First (and only) token
                
                token_activations.append({
                    "token": token,
                    "activation": float(activation),
                    "abs_activation": abs(float(activation))
                })
                
            except Exception as e:
                logger.warning(f"Failed to process token '{token}': {str(e)}")
                continue
        
        # Sort by absolute activation strength
        token_activations.sort(key=lambda x: x["abs_activation"], reverse=True)
        
        # Get top k results
        top_results = token_activations[:top_k]
        
        # Create result structure
        result = {
            "query_type": "neuron",
            "neuron_index": neuron_index,
            "layer_index": layer_index,
            "top_tokens": top_results,
            "total_tokens_tested": len(token_activations)
        }
        
        logger.info(f"Found {len(top_results)} top activating tokens for neuron {neuron_index}")
        return result
    
    def query_cluster_activations(self, cluster_indices: List[int], layer_index: int,
                                 top_k: int = 10) -> Dict[str, Any]:
        """
        Find tokens that most strongly activate a cluster of neurons.
        
        Args:
            cluster_indices: List of neuron indices in the cluster
            layer_index: Layer index of the neurons
            top_k: Number of top activating tokens to return
            
        Returns:
            Dictionary with query results
        """
        logger.info(f"Querying cluster with {len(cluster_indices)} neurons in layer {layer_index}")
        
        # Load model and create extractor if not already done
        if self.extractor is None:
            model, tokenizer = self.model_loader.load_model('gpt2')
            self.extractor = ActivationExtractor(model, tokenizer)
        
        # Test each token and collect average cluster activations
        token_activations = []
        
        for token in self.test_tokens:
            try:
                # Extract activations for this token
                activation_data = self.extractor.extract_activations(token)
                
                # Get activations for all neurons in the cluster
                layer_data = activation_data["layers"][layer_index]
                cluster_activations = []
                
                for neuron_idx in cluster_indices:
                    if neuron_idx < len(layer_data["neurons"]):
                        neuron_data = layer_data["neurons"][neuron_idx]
                        activation = neuron_data["activations"][0]
                        cluster_activations.append(float(activation))
                
                if cluster_activations:
                    # Calculate average activation for the cluster
                    avg_activation = np.mean(cluster_activations)
                    max_activation = max(cluster_activations)
                    
                    token_activations.append({
                        "token": token,
                        "avg_activation": float(avg_activation),
                        "max_activation": float(max_activation),
                        "abs_avg_activation": abs(float(avg_activation))
                    })
                
            except Exception as e:
                logger.warning(f"Failed to process token '{token}': {str(e)}")
                continue
        
        # Sort by absolute average activation strength
        token_activations.sort(key=lambda x: x["abs_avg_activation"], reverse=True)
        
        # Get top k results
        top_results = token_activations[:top_k]
        
        # Create result structure
        result = {
            "query_type": "cluster",
            "cluster_indices": cluster_indices,
            "layer_index": layer_index,
            "top_tokens": top_results,
            "total_tokens_tested": len(token_activations)
        }
        
        logger.info(f"Found {len(top_results)} top activating tokens for cluster")
        return result
    
    def save_query_result(self, query_result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save query result to JSON file.
        
        Args:
            query_result: Query result dictionary
            filename: Optional filename. If None, generates from query info.
            
        Returns:
            Path to saved file
        """
        if filename is None:
            # Generate filename from query info
            query_type = query_result["query_type"]
            if query_type == "neuron":
                neuron_idx = query_result["neuron_index"]
                layer_idx = query_result["layer_index"]
                filename = f"neuron_{layer_idx}_{neuron_idx}_query.json"
            else:  # cluster
                layer_idx = query_result["layer_index"]
                cluster_size = len(query_result["cluster_indices"])
                filename = f"cluster_{layer_idx}_{cluster_size}_neurons_query.json"
        
        return self.data_io.save_queries(query_result, filename)
    
    def load_query_result(self, filename: str) -> Dict[str, Any]:
        """
        Load query result from JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Query result dictionary
        """
        return self.data_io.load_queries(filename)
    
    def list_query_files(self) -> List[str]:
        """
        List all available query result files.
        
        Returns:
            List of query result filenames
        """
        return self.data_io.list_query_files() 