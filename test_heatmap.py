#!/usr/bin/env python3
"""
Test script for heatmap visualization functionality.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.utils.data_io import create_data_io
from backend.visualizations.heatmap import create_heatmap_visualizer

def test_heatmap_visualization():
    """Test heatmap visualization with existing activation data."""
    print("🧠 Testing NeuronScope Heatmap Visualization")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    data_io = create_data_io()
    heatmap_viz = create_heatmap_visualizer()
    
    # List available activation files
    activation_files = data_io.list_activation_files()
    if not activation_files:
        print("❌ No activation files found!")
        print("💡 Run test_activation.py first to generate activation data.")
        return False
    
    print(f"📁 Found {len(activation_files)} activation files")
    
    # Load the first activation file
    filename = activation_files[0]
    print(f"📖 Loading activation data from: {filename}")
    
    try:
        activation_data = data_io.load_activations(filename)
        print(f"✅ Loaded activation data:")
        print(f"   - Prompt: {activation_data['prompt']}")
        print(f"   - Tokens: {activation_data['tokens']}")
        print(f"   - Layers: {len(activation_data['layers'])}")
        
        # Create output directory for visualizations
        output_dir = Path("data/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test 1: Single layer heatmap
        print("\n🎨 Creating single layer heatmap...")
        layer_heatmap_path = output_dir / f"layer_0_heatmap_{Path(filename).stem}.png"
        heatmap_viz.create_layer_heatmap(
            activation_data, 
            layer_index=0, 
            max_neurons=100,
            save_path=str(layer_heatmap_path)
        )
        print(f"✅ Saved layer heatmap to: {layer_heatmap_path}")
        
        # Test 2: Multi-layer heatmap (first 3 layers)
        print("\n🎨 Creating multi-layer heatmap...")
        multi_layer_path = output_dir / f"multi_layer_heatmap_{Path(filename).stem}.png"
        heatmap_viz.create_multi_layer_heatmap(
            activation_data,
            layers_to_show=[0, 1, 2],  # First 3 layers
            max_neurons_per_layer=50,
            save_path=str(multi_layer_path)
        )
        print(f"✅ Saved multi-layer heatmap to: {multi_layer_path}")
        
        # Test 3: Activation summary
        print("\n🎨 Creating activation summary...")
        summary_path = output_dir / f"activation_summary_{Path(filename).stem}.png"
        heatmap_viz.create_activation_summary(
            activation_data,
            save_path=str(summary_path)
        )
        print(f"✅ Saved activation summary to: {summary_path}")
        
        print(f"\n🎉 All heatmap visualizations completed successfully!")
        print(f"📁 Visualizations saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Heatmap visualization test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_heatmap_visualization()
    if success:
        print("\n📝 Next steps:")
        print("1. Check the generated visualization files")
        print("2. Start building React frontend")
        print("3. Implement interactive visualizations")
    else:
        print("\n❌ Please fix the issues and try again.") 