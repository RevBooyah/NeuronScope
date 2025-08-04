#!/usr/bin/env python3
"""
Test script for NeuronScope Model Switching

This script tests the multi-model loading and switching functionality.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.models.gpt2_loader import MultiModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_model_switching():
    """Test model switching functionality."""
    print("🧠 Testing NeuronScope Model Switching")
    print("=" * 50)
    
    try:
        # Initialize model loader
        print("📦 Initializing multi-model loader...")
        model_loader = MultiModelLoader(use_quantization=True)
        
        # Get available models
        print("\n📋 Available models:")
        models = model_loader.get_available_models()
        recommended = model_loader.get_recommended_models()
        
        print(f"   Total models: {len(models)}")
        print(f"   Recommended models: {len(recommended)}")
        
        # Show recommended models
        print("\n⭐ Recommended models:")
        for name, config in recommended.items():
            print(f"   - {name}: {config['description']}")
        
        # Test loading a small model first (GPT-2)
        print("\n🔍 Testing GPT-2 model loading...")
        model, tokenizer = model_loader.load_model('gpt2')
        print(f"✅ Successfully loaded GPT-2")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Tokenizer type: {type(tokenizer).__name__}")
        print(f"   - Device: {next(model.parameters()).device}")
        
        # Get model info
        info = model_loader.get_model_info('gpt2')
        print(f"   - Layers: {info['layers']}")
        print(f"   - Hidden size: {info['hidden_size']}")
        print(f"   - Attention heads: {info['num_attention_heads']}")
        
        # Test memory usage
        memory = model_loader.get_memory_usage()
        print(f"\n💾 Memory usage:")
        for key, value in memory.items():
            print(f"   - {key}: {value}")
        
        # Test loading a different model (if available)
        print("\n🔄 Testing model switching...")
        
        # Try to load a small model that should be available
        test_models = ['microsoft/phi-2', 'google/gemma-2b', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0']
        
        for test_model in test_models:
            if test_model in models:
                print(f"\n🔍 Testing {test_model}...")
                try:
                    model2, tokenizer2 = model_loader.load_model(test_model)
                    print(f"✅ Successfully loaded {test_model}")
                    print(f"   - Model type: {type(model2).__name__}")
                    print(f"   - Device: {next(model2.parameters()).device}")
                    
                    # Get info
                    info2 = model_loader.get_model_info(test_model)
                    print(f"   - Layers: {info2['layers']}")
                    print(f"   - Size category: {info2['size_category']}")
                    
                    # Test memory usage after loading
                    memory2 = model_loader.get_memory_usage()
                    print(f"   - Memory after loading: {memory2.get('gpu_memory_allocated', 'N/A')}")
                    
                    break
                except Exception as e:
                    print(f"❌ Failed to load {test_model}: {str(e)}")
                    continue
        
        # Test getting models by size
        print("\n📊 Models by size category:")
        for size in ['tiny', 'small', 'medium', 'large']:
            size_models = model_loader.get_models_by_size(size)
            if size_models:
                print(f"   {size.upper()}: {len(size_models)} models")
                for name in list(size_models.keys())[:3]:  # Show first 3
                    print(f"     - {name}")
        
        print("\n🎉 Model switching tests completed successfully!")
        print("\n📝 Next steps:")
        print("1. Test the frontend model selector")
        print("2. Try switching between different model families")
        print("3. Test with larger models (if you have enough GPU memory)")
        print("4. Verify activation extraction works with different models")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        logger.exception("Test failed")
        return False
    
    return True

if __name__ == "__main__":
    success = test_model_switching()
    sys.exit(0 if success else 1) 