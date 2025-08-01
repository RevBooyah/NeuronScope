#!/usr/bin/env python3
"""
Simple test script for activation extraction.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.models.gpt2_loader import create_model_loader
from backend.activations.extractor import create_activation_extractor
from backend.utils.data_io import create_data_io

def test_single_prompt():
    """Test activation extraction with a single prompt."""
    print("🧠 Testing NeuronScope Activation Extraction")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    model_loader = create_model_loader()
    data_io = create_data_io()
    
    # Load GPT-2 small model
    print("🔧 Loading GPT-2 model...")
    model, tokenizer = model_loader.load_model('gpt2')
    
    # Create activation extractor
    extractor = create_activation_extractor(model, tokenizer)
    
    # Test with a simple prompt
    test_prompt = "Hello world"
    print(f"📝 Testing with prompt: '{test_prompt}'")
    
    # Extract activations
    activation_data = extractor.extract_activations(test_prompt)
    
    # Print some basic info
    print(f"✅ Successfully extracted activations!")
    print(f"   - Prompt: {activation_data['prompt']}")
    print(f"   - Tokens: {activation_data['tokens']}")
    print(f"   - Layers: {len(activation_data['layers'])}")
    print(f"   - Neurons per layer: {len(activation_data['layers'][0]['neurons'])}")
    
    # Calculate and print stats
    stats = extractor.get_activation_stats(activation_data)
    print(f"   - Mean activation: {stats['layer_stats'][0]['mean_activation']:.4f}")
    print(f"   - Activation range: [{stats['layer_stats'][0]['min_activation']:.4f}, {stats['layer_stats'][0]['max_activation']:.4f}]")
    
    # Save to file
    filename = data_io.save_activations(activation_data)
    print(f"💾 Saved to: {filename}")
    
    # Clean up
    model_loader.unload_model('gpt2')
    print("🧹 Cleaned up model")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_single_prompt() 