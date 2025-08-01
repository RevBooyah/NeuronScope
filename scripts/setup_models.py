#!/usr/bin/env python3
"""
CLI script for setting up GPT-2 models for NeuronScope.

This script helps users download and set up the required GPT-2 models
for activation extraction.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {
        'torch': False,
        'transformers': False,
        'numpy': False
    }
    
    try:
        import torch
        dependencies['torch'] = True
        logger.info(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        logger.warning("❌ PyTorch not available")
    
    try:
        import transformers
        dependencies['transformers'] = True
        logger.info(f"✅ Transformers {transformers.__version__} available")
    except ImportError:
        logger.warning("❌ Transformers not available")
    
    try:
        import numpy
        dependencies['numpy'] = True
        logger.info(f"✅ NumPy {numpy.__version__} available")
    except ImportError:
        logger.warning("❌ NumPy not available")
    
    return dependencies

def get_model_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available GPT-2 models."""
    return {
        'gpt2': {
            'name': 'gpt2',
            'layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'description': 'GPT-2 Small (124M parameters)',
            'download_size': '~500MB',
            'recommended': True
        },
        'gpt2-medium': {
            'name': 'gpt2-medium', 
            'layers': 24,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'description': 'GPT-2 Medium (355M parameters)',
            'download_size': '~1.4GB',
            'recommended': True
        },
        'gpt2-large': {
            'name': 'gpt2-large',
            'layers': 36,
            'hidden_size': 1280,
            'num_attention_heads': 20,
            'description': 'GPT-2 Large (774M parameters)',
            'download_size': '~3GB',
            'recommended': False
        }
    }

def print_setup_instructions():
    """Print setup instructions for users."""
    print("🧠 NeuronScope - Model Setup")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps = check_dependencies()
    
    if not all(deps.values()):
        print("\n❌ Missing dependencies detected!")
        print("\n📋 To install required dependencies, run:")
        print("   pip install torch transformers numpy pandas scikit-learn")
        print("\n💡 For GPU support, you may need:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("\n🔄 After installation, run this script again.")
        return False
    
    print("\n✅ All dependencies available!")
    
    # Show model information
    print("\n🤖 Available GPT-2 Models:")
    models = get_model_info()
    
    for model_name, info in models.items():
        status = "⭐" if info['recommended'] else "  "
        print(f"{status} {model_name}")
        print(f"    {info['description']}")
        print(f"    Layers: {info['layers']}, Hidden Size: {info['hidden_size']}")
        print(f"    Download Size: {info['download_size']}")
        print()
    
    print("💡 Recommended models for starting:")
    print("   - gpt2 (small, fast, good for testing)")
    print("   - gpt2-medium (balanced performance)")
    print("\n🚀 Ready to extract activations!")
    return True

def main():
    """Main function."""
    if not print_setup_instructions():
        sys.exit(1)
    
    print("\n📝 Next steps:")
    print("1. Run: python scripts/extract_activations.py")
    print("2. Check generated files in data/activations/")
    print("3. Start building the React frontend")

if __name__ == "__main__":
    main() 