#!/usr/bin/env python3
"""
Test script for NeuronScope Neuron Clustering

This script tests the clustering functionality to group neurons by
activation similarity.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.clustering.neuron_clustering import NeuronClusterer
from backend.utils.data_io import DataIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_clustering():
    """Test neuron clustering functionality."""
    print("🧠 Testing NeuronScope Neuron Clustering")
    print("=" * 50)
    
    try:
        # Initialize components
        print("📦 Initializing components...")
        data_io = DataIO()
        clusterer = NeuronClusterer(data_io=data_io)
        
        # Test clustering on layer 0
        print("\n🔍 Testing neuron clustering...")
        clustering_result = clusterer.cluster_neurons(
            layer_index=0,
            n_clusters=5,
            use_pca=True,
            pca_components=20
        )
        
        print(f"✅ Clustering completed!")
        print(f"   - Layer: {clustering_result['layer']}")
        print(f"   - Clusters: {clustering_result['n_clusters']}")
        print(f"   - Neurons: {clustering_result['n_neurons']}")
        print(f"   - Prompts: {clustering_result['n_prompts']}")
        
        # Show cluster information
        print("\n📊 Cluster information:")
        for cluster in clustering_result['clusters']:
            print(f"   Cluster {cluster['cluster_id']}: {cluster['size']} neurons")
        
        # Analyze cluster characteristics
        print("\n🔍 Analyzing cluster characteristics...")
        analysis = clusterer.analyze_cluster_characteristics(clustering_result)
        
        print("📊 Cluster analysis:")
        for cluster_analysis in analysis['cluster_analyses']:
            print(f"   Cluster {cluster_analysis['cluster_id']}:")
            print(f"     - Size: {cluster_analysis['size']} neurons")
            print(f"     - Mean activation magnitude: {cluster_analysis['mean_activation_magnitude']:.3f}")
            print(f"     - Activation variability: {cluster_analysis['activation_variability']:.3f}")
            print(f"     - Top characteristic prompts:")
            for prompt, z_score in cluster_analysis['top_characteristic_prompts'][:3]:
                print(f"       * '{prompt}' (z-score: {z_score:.2f})")
            print()
        
        # Save result
        filepath = clusterer.save_clustering_result(clustering_result)
        filename = filepath.split('/')[-1]  # Extract just the filename
        print(f"💾 Saved clustering result to: {filepath}")
        
        # Test loading
        print("\n📖 Testing result loading...")
        loaded_result = clusterer.load_clustering_result(filename)
        print(f"✅ Successfully loaded clustering result")
        print(f"   - Loaded {len(loaded_result['clusters'])} clusters")
        
        # List saved files
        print("\n📁 Available clustering files:")
        clustering_files = clusterer.list_clustering_files()
        for filename in clustering_files:
            print(f"   - {filename}")
        
        print("\n🎉 All clustering tests completed successfully!")
        print("\n📝 Next steps:")
        print("1. Check the generated clustering result files")
        print("2. Integrate with React frontend")
        print("3. Test clustering on different layers")
        print("4. Experiment with different numbers of clusters")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        logger.exception("Test failed")
        return False
    
    return True

if __name__ == "__main__":
    success = test_clustering()
    sys.exit(0 if success else 1) 