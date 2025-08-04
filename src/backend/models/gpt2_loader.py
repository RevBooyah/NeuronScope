"""
Multi-Model Loader for NeuronScope

This module handles loading and caching of various transformer models
for neuron activation extraction, including GPT-2, LLaMA, Mistral, and others.
"""

import os
import logging
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path

import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    MistralForCausalLM,
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModelLoader:
    """Handles loading and caching of various transformer models."""
    
    # Model configurations - organized by family
    MODEL_CONFIGS = {
        # GPT-2 Family (proven, reliable)
        'gpt2': {
            'name': 'gpt2',
            'family': 'gpt2',
            'layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'description': 'GPT-2 Small (124M parameters)',
            'model_class': GPT2LMHeadModel,
            'tokenizer_class': GPT2Tokenizer,
            'recommended': True,
            'size_category': 'small'
        },
        'gpt2-medium': {
            'name': 'gpt2-medium',
            'family': 'gpt2', 
            'layers': 24,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'description': 'GPT-2 Medium (355M parameters)',
            'model_class': GPT2LMHeadModel,
            'tokenizer_class': GPT2Tokenizer,
            'recommended': True,
            'size_category': 'medium'
        },
        'gpt2-large': {
            'name': 'gpt2-large',
            'family': 'gpt2',
            'layers': 36,
            'hidden_size': 1280,
            'num_attention_heads': 20,
            'description': 'GPT-2 Large (774M parameters)',
            'model_class': GPT2LMHeadModel,
            'tokenizer_class': GPT2Tokenizer,
            'recommended': False,
            'size_category': 'large'
        },
        
        # LLaMA 2 Family (excellent performance)
        'meta-llama/Llama-2-7b-hf': {
            'name': 'meta-llama/Llama-2-7b-hf',
            'family': 'llama2',
            'layers': 32,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'description': 'LLaMA 2 7B (7B parameters)',
            'model_class': LlamaForCausalLM,
            'tokenizer_class': LlamaTokenizer,
            'recommended': True,
            'size_category': 'medium',
            'requires_auth': True
        },
        'meta-llama/Llama-2-13b-hf': {
            'name': 'meta-llama/Llama-2-13b-hf',
            'family': 'llama2',
            'layers': 40,
            'hidden_size': 5120,
            'num_attention_heads': 40,
            'description': 'LLaMA 2 13B (13B parameters)',
            'model_class': LlamaForCausalLM,
            'tokenizer_class': LlamaTokenizer,
            'recommended': False,
            'size_category': 'large',
            'requires_auth': True
        },
        
        # Mistral Family (excellent performance, smaller size)
        'mistralai/Mistral-7B-v0.1': {
            'name': 'mistralai/Mistral-7B-v0.1',
            'family': 'mistral',
            'layers': 32,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'description': 'Mistral 7B v0.1 (7B parameters)',
            'model_class': MistralForCausalLM,
            'tokenizer_class': AutoTokenizer,
            'recommended': True,
            'size_category': 'medium'
        },
        'mistralai/Mistral-7B-Instruct-v0.2': {
            'name': 'mistralai/Mistral-7B-Instruct-v0.2',
            'family': 'mistral',
            'layers': 32,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'description': 'Mistral 7B Instruct v0.2 (7B parameters)',
            'model_class': MistralForCausalLM,
            'tokenizer_class': AutoTokenizer,
            'recommended': True,
            'size_category': 'medium'
        },
        
        # Microsoft Phi Family (small but powerful)
        'microsoft/phi-2': {
            'name': 'microsoft/phi-2',
            'family': 'phi',
            'layers': 32,
            'hidden_size': 2560,
            'num_attention_heads': 32,
            'description': 'Microsoft Phi-2 (2.7B parameters)',
            'model_class': AutoModelForCausalLM,
            'tokenizer_class': AutoTokenizer,
            'recommended': True,
            'size_category': 'small'
        },
        
        # Google Gemma Family
        'google/gemma-2b': {
            'name': 'google/gemma-2b',
            'family': 'gemma',
            'layers': 18,
            'hidden_size': 2048,
            'num_attention_heads': 8,
            'description': 'Google Gemma 2B (2B parameters)',
            'model_class': AutoModelForCausalLM,
            'tokenizer_class': AutoTokenizer,
            'recommended': True,
            'size_category': 'small'
        },
        'google/gemma-7b': {
            'name': 'google/gemma-7b',
            'family': 'gemma',
            'layers': 28,
            'hidden_size': 4096,
            'num_attention_heads': 16,
            'description': 'Google Gemma 7B (7B parameters)',
            'model_class': AutoModelForCausalLM,
            'tokenizer_class': AutoTokenizer,
            'recommended': False,
            'size_category': 'medium'
        },
        
        # TinyLlama (very small, good for testing)
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0': {
            'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'family': 'tinyllama',
            'layers': 22,
            'hidden_size': 2048,
            'num_attention_heads': 16,
            'description': 'TinyLlama 1.1B Chat (1.1B parameters)',
            'model_class': AutoModelForCausalLM,
            'tokenizer_class': AutoTokenizer,
            'recommended': True,
            'size_category': 'tiny'
        },
        
        # Qwen Family (Alibaba)
        'Qwen/Qwen-1_8B': {
            'name': 'Qwen/Qwen-1_8B',
            'family': 'qwen',
            'layers': 24,
            'hidden_size': 2048,
            'num_attention_heads': 16,
            'description': 'Qwen 1.8B (1.8B parameters)',
            'model_class': AutoModelForCausalLM,
            'tokenizer_class': AutoTokenizer,
            'recommended': True,
            'size_category': 'small'
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None, use_quantization: bool = False):
        """
        Initialize the multi-model loader.
        
        Args:
            cache_dir: Directory to cache downloaded models. If None, uses default.
            use_quantization: Whether to use 4-bit quantization for large models
        """
        self.cache_dir = cache_dir
        self.use_quantization = use_quantization
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models.
        
        Returns:
            Dictionary mapping model names to their configurations.
        """
        # Create a JSON-serializable copy without Python types
        serializable_configs = {}
        for name, config in self.MODEL_CONFIGS.items():
            serializable_config = config.copy()
            # Remove non-serializable fields
            serializable_config.pop('model_class', None)
            serializable_config.pop('tokenizer_class', None)
            serializable_configs[name] = serializable_config
        return serializable_configs
    
    def get_recommended_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get only the recommended models for easier selection.
        
        Returns:
            Dictionary of recommended models
        """
        # Create a JSON-serializable copy without Python types
        serializable_configs = {}
        for name, config in self.MODEL_CONFIGS.items():
            if config.get('recommended', False):
                serializable_config = config.copy()
                # Remove non-serializable fields
                serializable_config.pop('model_class', None)
                serializable_config.pop('tokenizer_class', None)
                serializable_configs[name] = serializable_config
        return serializable_configs
    
    def get_models_by_size(self, size_category: str) -> Dict[str, Dict[str, Any]]:
        """
        Get models filtered by size category.
        
        Args:
            size_category: 'tiny', 'small', 'medium', 'large'
            
        Returns:
            Dictionary of models in the specified size category
        """
        # Create a JSON-serializable copy without Python types
        serializable_configs = {}
        for name, config in self.MODEL_CONFIGS.items():
            if config.get('size_category') == size_category:
                serializable_config = config.copy()
                # Remove non-serializable fields
                serializable_config.pop('model_class', None)
                serializable_config.pop('tokenizer_class', None)
                serializable_configs[name] = serializable_config
        return serializable_configs
    
    def load_model(self, model_name: str = 'gpt2') -> Tuple[Any, Any]:
        """
        Load a model and tokenizer.
        
        Args:
            model_name: Name of the model to load
            
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
        
        config = self.MODEL_CONFIGS[model_name]
        
        try:
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Model info: {config['description']}")
            
            # Check if model requires authentication
            if config.get('requires_auth', False):
                logger.warning(f"Model {model_name} requires Hugging Face authentication")
                logger.info("Please set HF_TOKEN environment variable or login with huggingface-cli")
            
            # Setup quantization if requested and model is large
            quantization_config = None
            if self.use_quantization and config['size_category'] in ['medium', 'large']:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info(f"Using 4-bit quantization for {model_name}")
            
            # Load model with proper error handling
            model_class = config['model_class']
            
            # Determine if model needs trust_remote_code
            needs_trust_remote_code = model_name in [
                'Qwen/Qwen-1_8B', 'Qwen/Qwen-7B', 'Qwen/Qwen-14B',
                'microsoft/DialoGPT-medium', 'microsoft/DialoGPT-large'
            ]
            
            try:
                model = model_class.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if not self.use_quantization else torch.float32,
                    quantization_config=quantization_config,
                    output_hidden_states=True,  # Enable hidden states output
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=needs_trust_remote_code
                )
            except Exception as model_error:
                # If first attempt fails, try with trust_remote_code=True
                if not needs_trust_remote_code:
                    logger.info(f"Retrying {model_name} with trust_remote_code=True")
                    try:
                        model = model_class.from_pretrained(
                            model_name,
                            cache_dir=self.cache_dir,
                            torch_dtype=torch.float16 if not self.use_quantization else torch.float32,
                            quantization_config=quantization_config,
                            output_hidden_states=True,
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True
                        )
                    except Exception as retry_error:
                        raise RuntimeError(f"Failed to load model {model_name}: {str(retry_error)}")
                else:
                    raise RuntimeError(f"Failed to load model {model_name}: {str(model_error)}")
            
            # Load tokenizer with proper error handling
            tokenizer_class = config['tokenizer_class']
            
            try:
                tokenizer = tokenizer_class.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=needs_trust_remote_code
                )
            except Exception as tokenizer_error:
                # If first attempt fails, try with trust_remote_code=True
                if not needs_trust_remote_code:
                    logger.info(f"Retrying tokenizer for {model_name} with trust_remote_code=True")
                    try:
                        tokenizer = tokenizer_class.from_pretrained(
                            model_name,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True
                        )
                    except Exception as retry_error:
                        raise RuntimeError(f"Failed to load tokenizer for {model_name}: {str(retry_error)}")
                else:
                    raise RuntimeError(f"Failed to load tokenizer for {model_name}: {str(tokenizer_error)}")
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Move model to device if not using device_map
            if not torch.cuda.is_available() or quantization_config is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
            
            model.eval()  # Set to evaluation mode
            
            # Cache the loaded model
            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer
            
            device_info = str(next(model.parameters()).device)
            logger.info(f"Successfully loaded {model_name} on {device_info}")
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
        
        # Remove non-serializable fields
        info.pop('model_class', None)
        info.pop('tokenizer_class', None)
        
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
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information for loaded models.
        
        Returns:
            Dictionary with memory usage information
        """
        if not torch.cuda.is_available():
            return {"gpu_memory": "N/A (CUDA not available)"}
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_allocated = torch.cuda.memory_allocated(0)
        gpu_memory_reserved = torch.cuda.memory_reserved(0)
        
        return {
            "gpu_memory_total": f"{gpu_memory / 1024**3:.2f} GB",
            "gpu_memory_allocated": f"{gpu_memory_allocated / 1024**3:.2f} GB",
            "gpu_memory_reserved": f"{gpu_memory_reserved / 1024**3:.2f} GB",
            "gpu_memory_free": f"{(gpu_memory - gpu_memory_reserved) / 1024**3:.2f} GB"
        }


def create_model_loader(cache_dir: Optional[str] = None, use_quantization: bool = False) -> MultiModelLoader:
    """
    Factory function to create a MultiModelLoader instance.
    
    Args:
        cache_dir: Directory to cache downloaded models
        use_quantization: Whether to use quantization for large models
        
    Returns:
        MultiModelLoader instance
    """
    return MultiModelLoader(cache_dir=cache_dir, use_quantization=use_quantization)

# Backward compatibility
GPT2ModelLoader = MultiModelLoader 