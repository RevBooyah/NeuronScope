"""
GPT-2 Model Loader for NeuronScope

This module handles loading and caching of GPT-2 models (small, medium, large)
for neuron activation extraction.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2ModelLoader:
    """Handles loading and caching of GPT-2 models."""
    
    # Model configurations
    MODEL_CONFIGS = {
        'gpt2': {
            'name': 'gpt2',
            'layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'description': 'GPT-2 Small (124M parameters)'
        },
        'gpt2-medium': {
            'name': 'gpt2-medium', 
            'layers': 24,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'description': 'GPT-2 Medium (355M parameters)'
        },
        'gpt2-large': {
            'name': 'gpt2-large',
            'layers': 36,
            'hidden_size': 1280,
            'num_attention_heads': 20,
            'description': 'GPT-2 Large (774M parameters)'
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the GPT-2 model loader.
        
        Args:
            cache_dir: Directory to cache downloaded models. If None, uses default.
        """
        self.cache_dir = cache_dir
        self._models: Dict[str, GPT2LMHeadModel] = {}
        self._tokenizers: Dict[str, GPT2Tokenizer] = {}
        
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available GPT-2 models.
        
        Returns:
            Dictionary mapping model names to their configurations.
        """
        return self.MODEL_CONFIGS.copy()
    
    def load_model(self, model_name: str = 'gpt2') -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
        """
        Load a GPT-2 model and tokenizer.
        
        Args:
            model_name: Name of the model to load ('gpt2', 'gpt2-medium', 'gpt2-large')
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ValueError: If model_name is not supported
            RuntimeError: If model loading fails
        """
        if model_name not in self.MODEL_CONFIGS:
            available = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
        
        # Return cached model if already loaded
        if model_name in self._models:
            logger.info(f"Using cached model: {model_name}")
            return self._models[model_name], self._tokenizers[model_name]
        
        try:
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Model info: {self.MODEL_CONFIGS[model_name]['description']}")
            
            # Load model and tokenizer
            model = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,  # Use float32 for compatibility
                output_hidden_states=True   # Enable hidden states output
            )
            tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Move model to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()  # Set to evaluation mode
            
            # Cache the loaded model
            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded {model_name} on {device}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not supported")
        
        info = self.MODEL_CONFIGS[model_name].copy()
        
        # Add device info if model is loaded
        if model_name in self._models:
            model = self._models[model_name]
            info['device'] = str(next(model.parameters()).device)
            info['loaded'] = True
        else:
            info['loaded'] = False
            
        return info
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self._models:
            del self._models[model_name]
            del self._tokenizers[model_name]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Unloaded model: {model_name}")
    
    def unload_all_models(self) -> None:
        """Unload all cached models from memory."""
        model_names = list(self._models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        logger.info("Unloaded all models")


def create_model_loader(cache_dir: Optional[str] = None) -> GPT2ModelLoader:
    """
    Factory function to create a GPT2ModelLoader instance.
    
    Args:
        cache_dir: Directory to cache downloaded models
        
    Returns:
        GPT2ModelLoader instance
    """
    return GPT2ModelLoader(cache_dir=cache_dir) 