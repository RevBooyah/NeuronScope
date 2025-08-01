#!/usr/bin/env python3
"""
Test script for scatter plot visualization functionality.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.utils.data_io import create_data_io
from backend.visualizations.scatter import create_scatter_visualizer

def test_scatter_visualization():
    """Test scatter plot visualization with existing activation data."""
    print("🧠 Testing NeuronScope Scatter Plot Visualization")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    data_io = create_data_io()
    scatter_viz = create_scatter_visualizer()
    
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
        
        # Test 1: PCA scatter plot
        print("\n🎨 Creating PCA scatter plot...")
        pca_path = output_dir / f"pca_scatter_layer_0_{Path(filename).stem}.png"
        scatter_viz.create_pca_scatter(
            activation_data, 
            layer_index=0, 
            max_neurons=300,
            save_path=str(pca_path)
        )
        print(f"✅ Saved PCA scatter plot to: {pca_path}")
        
        # Test 2: t-SNE scatter plot (with fewer neurons for performance)
        print("\n🎨 Creating t-SNE scatter plot...")
        tsne_path = output_dir / f"tsne_scatter_layer_0_{Path(filename).stem}.png"
        scatter_viz.create_tsne_scatter(
            activation_data,
            layer_index=0,
            max_neurons=200,  # Fewer neurons for t-SNE performance
            perplexity=30.0,
            save_path=str(tsne_path)
        )
        print(f"✅ Saved t-SNE scatter plot to: {tsne_path}")
        
        # Test 3: PCA comparison across layers
        print("\n🎨 Creating PCA comparison across layers...")
        pca_comp_path = output_dir / f"pca_comparison_{Path(filename).stem}.png"
        scatter_viz.create_comparison_scatter(
            activation_data,
            layer_indices=[0, 1, 2],  # First 3 layers
            method='pca',
            max_neurons_per_layer=150,
            save_path=str(pca_comp_path)
        )
        print(f"✅ Saved PCA comparison to: {pca_comp_path}")
        
        print(f"\n🎉 All scatter plot visualizations completed successfully!")
        print(f"📁 Visualizations saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scatter plot visualization test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_scatter_visualization()
    if success:
        print("\n📝 Next steps:")
        print("1. Check the generated scatter plot files")
        print("2. Start building React frontend")
        print("3. Implement interactive visualizations")
    else:
        print("\n❌ Please fix the issues and try again.") 