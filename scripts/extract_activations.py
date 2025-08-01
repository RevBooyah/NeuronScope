#!/usr/bin/env python3
"""
CLI script for extracting neuron activations from GPT-2 models.

This script demonstrates the activation extraction functionality using
sample prompts and saves the results to JSON files.
"""

import sys
import json
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.models.gpt2_loader import create_model_loader
from backend.activations.extractor import create_activation_extractor
from backend.utils.data_io import create_data_io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_prompts() -> list[str]:
    """Load sample prompts from samples.json."""
    try:
        with open("samples.json", "r") as f:
            samples = json.load(f)
        return [prompt for prompt in samples if prompt.strip()]  # Filter empty prompts
    except FileNotFoundError:
        logger.warning("samples.json not found, using default prompts")
        return [
            "Hello world",
            "What is the capital of France?",
            "The cat sat on the mat."
        ]

def main():
    """Main function for activation extraction."""
    print("🧠 NeuronScope - Activation Extraction")
    print("=" * 50)
    
    # Load sample prompts
    prompts = load_sample_prompts()
    print(f"Loaded {len(prompts)} sample prompts")
    
    # Initialize components
    print("\n📦 Initializing components...")
    model_loader = create_model_loader()
    data_io = create_data_io()
    
    # Get available models
    available_models = model_loader.get_available_models()
    print(f"Available models: {list(available_models.keys())}")
    
    # Process each model
    for model_name in ['gpt2', 'gpt2-medium']:  # Start with smaller models
        print(f"\n🔧 Processing model: {model_name}")
        print(f"Model info: {available_models[model_name]['description']}")
        
        try:
            # Load model
            model, tokenizer = model_loader.load_model(model_name)
            
            # Create activation extractor
            extractor = create_activation_extractor(model, tokenizer)
            
            # Process each prompt
            for i, prompt in enumerate(prompts, 1):
                print(f"\n  📝 Processing prompt {i}/{len(prompts)}: '{prompt[:50]}...'")
                
                try:
                    # Extract activations
                    activation_data = extractor.extract_activations(prompt)
                    
                    # Calculate stats
                    stats = extractor.get_activation_stats(activation_data)
                    print(f"    ✅ Extracted {stats['num_layers']} layers, {stats['layer_stats'][0]['num_neurons']} neurons per layer")
                    
                    # Save to file
                    filename = data_io.save_activations(activation_data)
                    print(f"    💾 Saved to: {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to process prompt '{prompt}': {str(e)}")
                    continue
            
            # Unload model to free memory
            model_loader.unload_model(model_name)
            print(f"  🧹 Unloaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {str(e)}")
            continue
    
    print("\n✅ Activation extraction completed!")
    print(f"📁 Check the 'data/activations/' directory for results")

if __name__ == "__main__":
    main() 