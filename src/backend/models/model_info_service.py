"""
Model Information Service for NeuronScope

This module provides comprehensive information about GPT-2 models including
architecture details, activation equations, training information, and model statistics.
"""

import torch
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

class ModelInfoService:
    """Service for providing comprehensive model information."""
    
    # GPT-2 Architecture Details
    GPT2_ARCHITECTURE = {
        'model_type': 'GPT-2 (Generative Pre-trained Transformer 2)',
        'architecture': 'Transformer Decoder-Only',
        'training_objective': 'Language Modeling (Next Token Prediction)',
        'training_data': 'WebText dataset (~40GB of text from Reddit)',
        'training_method': 'Unsupervised pre-training with causal language modeling',
        'vocabulary_size': 50257,
        'position_encoding': 'Learned positional embeddings',
        'activation_function': 'GELU (Gaussian Error Linear Unit)',
        'layer_norm': 'Pre-norm (before attention and MLP)',
        'attention_type': 'Multi-head self-attention with causal masking',
        'optimizer': 'Adam with weight decay',
        'learning_rate': '2.5e-4 with cosine annealing',
        'batch_size': '512 sequences',
        'context_length': 1024,
        'training_steps': 'Unknown (paper doesn\'t specify)',
        'hardware': '8 P100 GPUs',
        'training_time': 'Unknown (paper doesn\'t specify)',
        'paper': 'Language Models are Unsupervised Multitask Learners (Radford et al., 2019)',
        'license': 'MIT License',
        'organization': 'OpenAI'
    }
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'gpt2': {
            'name': 'gpt2',
            'layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'max_position_embeddings': 1024,
            'vocab_size': 50257,
            'parameters': 124_439_808,
            'description': 'GPT-2 Small (124M parameters)',
            'download_size': '~500MB',
            'recommended': True,
            'memory_usage': '~500MB RAM',
            'inference_speed': '~50ms per token (CPU)',
            'training_data_size': '~40GB',
            'release_date': 'February 2019'
        },
        'gpt2-medium': {
            'name': 'gpt2-medium', 
            'layers': 24,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'max_position_embeddings': 1024,
            'vocab_size': 50257,
            'parameters': 355_124_736,
            'description': 'GPT-2 Medium (355M parameters)',
            'download_size': '~1.4GB',
            'recommended': True,
            'memory_usage': '~1.4GB RAM',
            'inference_speed': '~150ms per token (CPU)',
            'training_data_size': '~40GB',
            'release_date': 'February 2019'
        },
        'gpt2-large': {
            'name': 'gpt2-large',
            'layers': 36,
            'hidden_size': 1280,
            'num_attention_heads': 20,
            'intermediate_size': 5120,
            'max_position_embeddings': 1024,
            'vocab_size': 50257,
            'parameters': 774_030_080,
            'description': 'GPT-2 Large (774M parameters)',
            'download_size': '~3GB',
            'recommended': False,
            'memory_usage': '~3GB RAM',
            'inference_speed': '~400ms per token (CPU)',
            'training_data_size': '~40GB',
            'release_date': 'February 2019'
        }
    }
    
    def __init__(self):
        """Initialize the model information service."""
        self._loaded_models: Dict[str, GPT2LMHeadModel] = {}
        self._loaded_tokenizers: Dict[str, GPT2Tokenizer] = {}
    
    def get_comprehensive_model_info(self, model_name: str = 'gpt2') -> Dict[str, Any]:
        """
        Get comprehensive information about a specific model.
        
        Args:
            model_name: Name of the model ('gpt2', 'gpt2-medium', 'gpt2-large')
            
        Returns:
            Dictionary with comprehensive model information
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not supported")
        
        config = self.MODEL_CONFIGS[model_name].copy()
        
        # Add architecture information
        info = {
            'basic_info': config,
            'architecture': self._get_architecture_info(model_name),
            'activation_equations': self._get_activation_equations(),
            'training_details': self._get_training_details(),
            'layer_details': self._get_layer_details(model_name),
            'attention_mechanism': self._get_attention_mechanism_info(),
            'tokenization': self._get_tokenization_info(),
            'performance_metrics': self._get_performance_metrics(model_name),
            'usage_guidelines': self._get_usage_guidelines(model_name)
        }
        
        # Add runtime information if model is loaded
        if model_name in self._loaded_models:
            info['runtime_info'] = self._get_runtime_info(model_name)
        
        return info
    
    def _get_architecture_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed architecture information."""
        config = self.MODEL_CONFIGS[model_name]
        
        return {
            'overview': self.GPT2_ARCHITECTURE,
            'dimensions': {
                'embedding_dimension': config['hidden_size'],
                'attention_heads': config['num_attention_heads'],
                'head_dimension': config['hidden_size'] // config['num_attention_heads'],
                'mlp_intermediate_size': config['intermediate_size'],
                'total_layers': config['layers'],
                'vocabulary_size': config['vocab_size'],
                'max_sequence_length': config['max_position_embeddings']
            },
            'parameter_distribution': self._calculate_parameter_distribution(config),
            'memory_layout': self._get_memory_layout_info(config)
        }
    
    def _get_activation_equations(self) -> Dict[str, Any]:
        """Get mathematical equations for model components."""
        return {
            'attention_mechanism': {
                'query_key_value': 'Q = XW_q, K = XW_k, V = XW_v',
                'attention_scores': 'Attention(Q,K,V) = softmax(QK^T/√d_k)V',
                'multi_head': 'MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O',
                'where': 'head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)'
            },
            'feed_forward': {
                'mlp': 'FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2',
                'gelu': 'GELU(x) = x * Φ(x) where Φ is the CDF of N(0,1)',
                'approximation': 'GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))'
            },
            'layer_norm': {
                'normalization': 'LayerNorm(x) = γ * (x - μ)/√(σ² + ε) + β',
                'where': 'μ = mean(x), σ² = var(x), γ,β are learnable parameters'
            },
            'residual_connections': {
                'attention_residual': 'x\' = x + MultiHeadAttention(LayerNorm(x))',
                'mlp_residual': 'x\'\' = x\' + FFN(LayerNorm(x\'))'
            },
            'position_encoding': {
                'learned': 'PE(pos, 2i) = sin(pos/10000^(2i/d_model))',
                'learned_cos': 'PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))'
            }
        }
    
    def _get_training_details(self) -> Dict[str, Any]:
        """Get training information and methodology."""
        return {
            'dataset': {
                'name': 'WebText',
                'size': '~40GB of text data',
                'source': 'Reddit posts with >3 karma',
                'preprocessing': 'Byte-pair encoding (BPE) tokenization',
                'vocabulary': '50,257 tokens'
            },
            'training_objective': {
                'loss_function': 'Cross-entropy loss',
                'objective': 'Next token prediction (causal language modeling)',
                'formula': 'L = -∑(t=1 to T) log P(x_t | x_<t)',
                'where': 'x_t is the t-th token, x_<t are previous tokens'
            },
            'optimization': {
                'optimizer': 'Adam',
                'learning_rate': '2.5e-4',
                'scheduling': 'Cosine annealing',
                'weight_decay': '0.01',
                'gradient_clipping': '1.0',
                'batch_size': '512 sequences',
                'context_length': '1024 tokens'
            },
            'regularization': {
                'dropout': '0.1 (attention and MLP layers)',
                'weight_decay': '0.01',
                'layer_norm': 'Pre-norm for training stability'
            },
            'hardware': {
                'gpus': '8 P100 GPUs',
                'memory_per_gpu': '16GB',
                'distributed_training': 'Data parallel across GPUs'
            }
        }
    
    def _get_layer_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about model layers."""
        config = self.MODEL_CONFIGS[model_name]
        
        return {
            'layer_structure': {
                'total_layers': config['layers'],
                'layer_types': ['Transformer Block'] * config['layers'],
                'layer_components': [
                    'Layer Normalization',
                    'Multi-Head Self-Attention',
                    'Residual Connection',
                    'Layer Normalization',
                    'Feed-Forward Network (MLP)',
                    'Residual Connection'
                ]
            },
            'attention_details': {
                'num_heads': config['num_attention_heads'],
                'head_dimension': config['hidden_size'] // config['num_attention_heads'],
                'attention_pattern': 'Causal (lower triangular)',
                'attention_mechanism': 'Scaled dot-product attention',
                'attention_dropout': '0.1'
            },
            'mlp_details': {
                'input_dimension': config['hidden_size'],
                'hidden_dimension': config['intermediate_size'],
                'output_dimension': config['hidden_size'],
                'activation': 'GELU',
                'dropout': '0.1'
            },
            'embedding_layers': {
                'token_embeddings': f"{config['vocab_size']} × {config['hidden_size']}",
                'position_embeddings': f"{config['max_position_embeddings']} × {config['hidden_size']}",
                'embedding_dropout': '0.1'
            }
        }
    
    def _get_attention_mechanism_info(self) -> Dict[str, Any]:
        """Get detailed attention mechanism information."""
        return {
            'mechanism_type': 'Multi-Head Self-Attention',
            'mathematical_formulation': {
                'attention_scores': 'A = QK^T/√d_k',
                'attention_weights': 'W = softmax(A)',
                'output': 'O = WV',
                'multi_head': 'MultiHead = Concat(head_1,...,head_h)W^O'
            },
            'causal_masking': {
                'purpose': 'Prevent attending to future tokens',
                'implementation': 'Add large negative values to future positions',
                'mask': 'Lower triangular matrix (1s below diagonal, 0s above)'
            },
            'attention_patterns': {
                'local_attention': 'Tokens attend to nearby tokens',
                'global_attention': 'Some tokens can attend to all positions',
                'sparse_attention': 'Not used in GPT-2 (dense attention)'
            },
            'attention_analysis': {
                'attention_heads': 'Different heads capture different patterns',
                'attention_visualization': 'Heatmaps showing attention weights',
                'interpretability': 'Attention weights can be analyzed for insights'
            }
        }
    
    def _get_tokenization_info(self) -> Dict[str, Any]:
        """Get tokenization information."""
        return {
            'tokenizer_type': 'Byte-Pair Encoding (BPE)',
            'vocabulary': {
                'size': 50257,
                'special_tokens': [
                    '<|endoftext|>',
                    '<|unk|>',
                    '<|pad|>'
                ],
            },
            'bpe_algorithm': {
                'description': 'Subword tokenization that merges frequent character pairs',
                'merges': 'Learned from training data',
                'vocabulary': 'Built incrementally during training'
            },
            'tokenization_process': {
                'step1': 'Split text into characters',
                'step2': 'Apply BPE merges in order',
                'step3': 'Convert to token IDs',
                'step4': 'Add special tokens as needed'
            },
            'handling_unknown': {
                'unknown_token': '<|unk|>',
                'fallback': 'Character-level tokenization for unknown words'
            }
        }
    
    def _get_performance_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics and benchmarks."""
        config = self.MODEL_CONFIGS[model_name]
        
        return {
            'computational_complexity': {
                'time_complexity': f"O(n²d) where n=sequence_length, d=hidden_size",
                'space_complexity': f"O(n²) for attention matrices",
                'parameters': f"{config['parameters']:,}",
                'memory_usage': config['memory_usage']
            },
            'inference_speed': {
                'cpu_speed': config['inference_speed'],
                'gpu_speed': f"~{int(config['inference_speed'].split('~')[1].split('ms')[0])//10}ms per token (GPU)",
                'throughput': f"~{1000//int(config['inference_speed'].split('~')[1].split('ms')[0])} tokens/second (CPU)"
            },
            'model_size': {
                'parameters': f"{config['parameters']:,}",
                'download_size': config['download_size'],
                'memory_footprint': config['memory_usage']
            },
            'quality_metrics': {
                'perplexity': 'Varies by dataset and task',
                'accuracy': 'Task-dependent',
                'bleu_score': 'For translation tasks',
                'rouge_score': 'For summarization tasks'
            }
        }
    
    def _get_usage_guidelines(self, model_name: str) -> Dict[str, Any]:
        """Get usage guidelines and best practices."""
        return {
            'recommended_use_cases': [
                'Text generation',
                'Language modeling',
                'Text completion',
                'Creative writing',
                'Code generation',
                'Question answering (with fine-tuning)'
            ],
            'limitations': [
                'No factual knowledge cutoff date',
                'May generate biased or harmful content',
                'Limited context window (1024 tokens)',
                'No built-in safety mechanisms',
                'Can hallucinate information'
            ],
            'best_practices': [
                'Use appropriate temperature and top-k/top-p sampling',
                'Implement content filtering',
                'Validate generated content',
                'Use few-shot prompting for better results',
                'Consider fine-tuning for specific tasks'
            ],
            'safety_considerations': [
                'Implement content moderation',
                'Use safety classifiers',
                'Monitor for harmful outputs',
                'Provide clear usage guidelines',
                'Consider ethical implications'
            ]
        }
    
    def _calculate_parameter_distribution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate parameter distribution across model components."""
        total_params = config['parameters']
        hidden_size = config['hidden_size']
        layers = config['layers']
        vocab_size = config['vocab_size']
        intermediate_size = config['intermediate_size']
        num_heads = config['num_attention_heads']
        head_dim = hidden_size // num_heads
        
        # Approximate parameter distribution
        embedding_params = vocab_size * hidden_size + config['max_position_embeddings'] * hidden_size
        
        # Per layer parameters
        attention_params_per_layer = (
            3 * hidden_size * hidden_size +  # Q, K, V projections
            hidden_size * hidden_size +      # Output projection
            4 * hidden_size                  # Layer norms
        )
        
        mlp_params_per_layer = (
            hidden_size * intermediate_size + intermediate_size +  # First layer
            intermediate_size * hidden_size + hidden_size +        # Second layer
            2 * hidden_size                                         # Layer norms
        )
        
        total_layer_params = (attention_params_per_layer + mlp_params_per_layer) * layers
        lm_head_params = vocab_size * hidden_size
        
        return {
            'embeddings': {
                'parameters': embedding_params,
                'percentage': round(embedding_params / total_params * 100, 1)
            },
            'transformer_layers': {
                'parameters': total_layer_params,
                'percentage': round(total_layer_params / total_params * 100, 1),
                'per_layer': attention_params_per_layer + mlp_params_per_layer
            },
            'language_model_head': {
                'parameters': lm_head_params,
                'percentage': round(lm_head_params / total_params * 100, 1)
            },
            'total': {
                'parameters': total_params,
                'percentage': 100.0
            }
        }
    
    def _get_memory_layout_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get memory layout and storage information."""
        return {
            'parameter_storage': {
                'precision': '32-bit float (FP32)',
                'bytes_per_parameter': 4,
                'total_memory': f"{config['parameters'] * 4 / (1024**3):.1f} GB",
                'activation_memory': 'Additional memory for forward pass activations'
            },
            'attention_memory': {
                'attention_matrix_size': f"O(sequence_length² × num_heads)",
                'memory_per_head': 'sequence_length × sequence_length × head_dimension',
                'total_attention_memory': 'Varies with sequence length'
            },
            'gradient_memory': {
                'training_memory': '2x parameter memory (for gradients)',
                'optimizer_memory': 'Additional memory for optimizer states'
            }
        }
    
    def _get_runtime_info(self, model_name: str) -> Dict[str, Any]:
        """Get runtime information for loaded models."""
        if model_name not in self._loaded_models:
            return {'loaded': False}
        
        model = self._loaded_models[model_name]
        
        return {
            'loaded': True,
            'device': str(next(model.parameters()).device),
            'dtype': str(next(model.parameters()).dtype),
            'model_state': 'eval' if model.training else 'training',
            'memory_allocated': f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB" if torch.cuda.is_available() else 'N/A',
            'memory_reserved': f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB" if torch.cuda.is_available() else 'N/A'
        }
    
    def register_loaded_model(self, model_name: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
        """Register a loaded model for runtime information."""
        self._loaded_models[model_name] = model
        self._loaded_tokenizers[model_name] = tokenizer
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        return {
            model_name: self.get_comprehensive_model_info(model_name)
            for model_name in self.MODEL_CONFIGS.keys()
        }

# Global instance
model_info_service = ModelInfoService() 