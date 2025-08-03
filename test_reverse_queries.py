#!/usr/bin/env python3
"""
Test script for NeuronScope Reverse Activation Queries

This script tests the reverse query functionality to find tokens that
strongly activate specific neurons.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.queries.reverse_queries import ReverseQueryEngine
from backend.utils.data_io import DataIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_reverse_queries():
    """Test reverse activation queries functionality."""
    print("🧠 Testing NeuronScope Reverse Activation Queries")
    print("=" * 50)
    
    try:
        # Initialize components
        print("📦 Initializing components...")
        data_io = DataIO()
        query_engine = ReverseQueryEngine(data_io=data_io)
        
        # Test neuron query
        print("\n🔍 Testing neuron query...")
        neuron_result = query_engine.query_neuron_activations(
            neuron_index=42, 
            layer_index=0, 
            top_k=5
        )
        
        print(f"✅ Neuron query completed!")
        print(f"   - Neuron: {neuron_result['neuron_index']} in layer {neuron_result['layer_index']}")
        print(f"   - Top tokens found: {len(neuron_result['top_tokens'])}")
        print(f"   - Tokens tested: {neuron_result['total_tokens_tested']}")
        
        # Show top results
        print("\n📊 Top activating tokens:")
        for i, token_data in enumerate(neuron_result['top_tokens'][:3]):
            print(f"   {i+1}. '{token_data['token']}' (activation: {token_data['activation']:.3f})")
        
        # Save result
        filename = query_engine.save_query_result(neuron_result)
        print(f"\n💾 Saved query result to: {filename}")
        
        # Test cluster query
        print("\n🔍 Testing cluster query...")
        cluster_result = query_engine.query_cluster_activations(
            cluster_indices=[10, 20, 30, 40, 50],
            layer_index=0,
            top_k=5
        )
        
        print(f"✅ Cluster query completed!")
        print(f"   - Cluster size: {len(cluster_result['cluster_indices'])} neurons")
        print(f"   - Layer: {cluster_result['layer_index']}")
        print(f"   - Top tokens found: {len(cluster_result['top_tokens'])}")
        
        # Show top results
        print("\n📊 Top activating tokens for cluster:")
        for i, token_data in enumerate(cluster_result['top_tokens'][:3]):
            print(f"   {i+1}. '{token_data['token']}' (avg: {token_data['avg_activation']:.3f}, max: {token_data['max_activation']:.3f})")
        
        # Save result
        filename = query_engine.save_query_result(cluster_result)
        print(f"\n💾 Saved cluster query result to: {filename}")
        
        # List saved files
        print("\n📁 Available query files:")
        query_files = query_engine.list_query_files()
        for filename in query_files:
            print(f"   - {filename}")
        
        print("\n🎉 All reverse query tests completed successfully!")
        print("\n📝 Next steps:")
        print("1. Check the generated query result files")
        print("2. Integrate with React frontend")
        print("3. Add more sophisticated token sets")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        logger.exception("Test failed")
        return False
    
    return True

if __name__ == "__main__":
    success = test_reverse_queries()
    sys.exit(0 if success else 1) 