"""
Neuron Activation Extractor for NeuronScope

This module extracts neuron activations from GPT-2 models and formats them
according to the defined JSON structure in DATA_STRUCTURE.md.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

class ActivationExtractor:
    """Extracts neuron activations from GPT-2 models."""
    
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
        """
        Initialize the activation extractor.
        
        Args:
            model: Loaded GPT-2 model
            tokenizer: GPT-2 tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def extract_activations(self, prompt: str) -> Dict[str, Any]:
        """
        Extract neuron activations for a given prompt.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary containing activation data in the format defined in DATA_STRUCTURE.md
        """
        try:
            # Tokenize the prompt
            tokens = self.tokenizer.tokenize(prompt)
            token_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            logger.info(f"Processing prompt: '{prompt}'")
            logger.info(f"Tokenized into {len(tokens)} tokens: {tokens}")
            
            # Get model outputs with hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=token_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Extract hidden states (activations)
            hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
            
            # Format activations according to DATA_STRUCTURE.md
            activation_data = {
                "prompt": prompt,
                "tokens": tokens,
                "layers": []
            }
            
            # Process each layer (skip the first hidden state as it's the input embeddings)
            for layer_idx, hidden_state in enumerate(hidden_states[1:], start=0):
                # hidden_state shape: (batch_size, sequence_length, hidden_size)
                layer_activations = hidden_state[0]  # Remove batch dimension
                
                layer_data = {
                    "layer_index": layer_idx,
                    "neurons": []
                }
                
                # Process each neuron in the layer
                for neuron_idx in range(layer_activations.shape[1]):
                    neuron_activations = layer_activations[:, neuron_idx].cpu().numpy()
                    
                    neuron_data = {
                        "neuron_index": neuron_idx,
                        "activations": neuron_activations.tolist()
                    }
                    
                    layer_data["neurons"].append(neuron_data)
                
                activation_data["layers"].append(layer_data)
            
            logger.info(f"Extracted activations for {len(activation_data['layers'])} layers")
            return activation_data
            
        except Exception as e:
            logger.error(f"Failed to extract activations for prompt '{prompt}': {str(e)}")
            raise RuntimeError(f"Activation extraction failed: {str(e)}")
    
    def extract_activations_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract activations for multiple prompts.
        
        Args:
            prompts: List of input text prompts
            
        Returns:
            List of activation data dictionaries
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            try:
                activation_data = self.extract_activations(prompt)
                results.append(activation_data)
            except Exception as e:
                logger.error(f"Failed to process prompt {i+1}: {str(e)}")
                # Continue with other prompts
                continue
        
        return results
    
    def get_activation_stats(self, activation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics for activation data.
        
        Args:
            activation_data: Activation data dictionary
            
        Returns:
            Dictionary containing activation statistics
        """
        stats = {
            "prompt": activation_data["prompt"],
            "num_tokens": len(activation_data["tokens"]),
            "num_layers": len(activation_data["layers"]),
            "layer_stats": []
        }
        
        for layer_data in activation_data["layers"]:
            layer_idx = layer_data["layer_index"]
            neurons = layer_data["neurons"]
            
            # Calculate statistics for this layer
            all_activations = []
            for neuron in neurons:
                all_activations.extend(neuron["activations"])
            
            all_activations = np.array(all_activations)
            
            layer_stats = {
                "layer_index": layer_idx,
                "num_neurons": len(neurons),
                "mean_activation": float(np.mean(all_activations)),
                "std_activation": float(np.std(all_activations)),
                "min_activation": float(np.min(all_activations)),
                "max_activation": float(np.max(all_activations)),
                "sparsity": float(np.mean(all_activations == 0))
            }
            
            stats["layer_stats"].append(layer_stats)
        
        return stats


def create_activation_extractor(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> ActivationExtractor:
    """
    Factory function to create an ActivationExtractor instance.
    
    Args:
        model: Loaded GPT-2 model
        tokenizer: GPT-2 tokenizer
        
    Returns:
        ActivationExtractor instance
    """
    return ActivationExtractor(model, tokenizer) 