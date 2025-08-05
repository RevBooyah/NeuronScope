#!/usr/bin/env python3
"""
Test script for pruning analysis functionality

This script tests the weight analysis and pruning impact analysis features
to ensure they work correctly with the NeuronScope backend.
"""

import sys
import json
import requests
import time
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / 'src' / 'backend'
sys.path.insert(0, str(backend_dir))

def test_api_server():
    """Test the pruning analysis API endpoints."""
    base_url = "http://localhost:5001/api"
    
    print("🧠 Testing NeuronScope Pruning Analysis API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Is it running?")
        return False
    
    # Test 2: Weight analysis
    print("\n2. Testing weight analysis...")
    try:
        response = requests.get(f"{base_url}/pruning/weight-analysis")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Weight analysis successful")
            print(f"   - Overall sparsity: {data.get('overall_sparsity', 0):.3f}")
            print(f"   - Total parameters: {data.get('total_parameters', 0):,}")
            print(f"   - Non-zero parameters: {data.get('non_zero_parameters', 0):,}")
        else:
            print(f"❌ Weight analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Weight analysis error: {str(e)}")
        return False
    
    # Test 3: Pruning candidates
    print("\n3. Testing pruning candidates...")
    try:
        response = requests.get(f"{base_url}/pruning/candidates?threshold=10.0")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pruning candidates analysis successful")
            print(f"   - Total candidates: {data.get('total_candidates', 0)}")
            print(f"   - Threshold percentile: {data.get('threshold_percentile', 0)}%")
            
            candidates = data.get('candidates', [])
            if candidates:
                print(f"   - Sample candidate: Layer {candidates[0]['layer_index']}, "
                      f"Neuron {candidates[0]['neuron_index']}, "
                      f"Score: {candidates[0]['pruning_score']:.3f}")
        else:
            print(f"❌ Pruning candidates failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Pruning candidates error: {str(e)}")
        return False
    
    # Test 4: Pruning impact analysis
    print("\n4. Testing pruning impact analysis...")
    try:
        test_data = {
            "layer_index": 0,
            "neuron_indices": [0, 1, 2],
            "input_text": "Hello world, this is a test for pruning analysis."
        }
        
        response = requests.post(f"{base_url}/pruning/impact-analysis", 
                               json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pruning impact analysis successful")
            print(f"   - Impact score: {data.get('impact_score', 0):.3f}")
            print(f"   - Mean change: {data.get('mean_change', 0):.3f}")
            print(f"   - Max change: {data.get('max_change', 0):.3f}")
            print(f"   - Safe to prune: {data.get('safe_to_prune', False)}")
        else:
            print(f"❌ Pruning impact analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Pruning impact analysis error: {str(e)}")
        return False
    
    # Test 5: Neuron importance analysis
    print("\n5. Testing neuron importance analysis...")
    try:
        test_data = {
            "layer_index": 0,
            "input_texts": [
                "Hello world",
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is fascinating"
            ]
        }
        
        response = requests.post(f"{base_url}/pruning/neuron-importance", 
                               json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Neuron importance analysis successful")
            print(f"   - Total neurons: {data.get('total_neurons', 0)}")
            print(f"   - Critical neurons: {data.get('critical_neurons', 0)}")
            
            importance_scores = data.get('importance_scores', [])
            if importance_scores:
                print(f"   - Sample neuron: Index {importance_scores[0]['neuron_index']}, "
                      f"Impact: {importance_scores[0]['impact_score']:.3f}, "
                      f"Critical: {importance_scores[0]['is_critical']}")
        else:
            print(f"❌ Neuron importance analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Neuron importance analysis error: {str(e)}")
        return False
    
    # Test 6: Batch pruning analysis
    print("\n6. Testing batch pruning analysis...")
    try:
        test_data = {
            "pruning_candidates": [
                {"layer_index": 0, "neuron_index": 0},
                {"layer_index": 0, "neuron_index": 1},
                {"layer_index": 1, "neuron_index": 0}
            ],
            "input_texts": ["Hello world", "Test input"],
            "batch_size": 5
        }
        
        response = requests.post(f"{base_url}/pruning/batch-analysis", 
                               json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch pruning analysis successful")
            print(f"   - Individual impacts: {len(data.get('individual_impacts', []))}")
            
            recommendations = data.get('recommendations', {})
            print(f"   - Safe to prune: {recommendations.get('safe_to_prune', 0)}")
            print(f"   - Risky to prune: {recommendations.get('risky_to_prune', 0)}")
            print(f"   - Suggested batch size: {recommendations.get('suggested_batch_size', 0)}")
        else:
            print(f"❌ Batch pruning analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch pruning analysis error: {str(e)}")
        return False
    
    # Test 7: Export functionality
    print("\n7. Testing export functionality...")
    try:
        test_data = {"type": "weight"}
        
        response = requests.post(f"{base_url}/pruning/export", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Export functionality successful")
            print(f"   - File path: {data.get('file_path', 'N/A')}")
        else:
            print(f"❌ Export functionality failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Export functionality error: {str(e)}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All pruning analysis tests passed!")
    return True

def test_direct_module():
    """Test the pruning analysis modules directly."""
    print("\n🔧 Testing pruning analysis modules directly")
    print("=" * 50)
    
    try:
        from models.gpt2_loader import MultiModelLoader
        from pruning import WeightAnalyzer, PruningImpactAnalyzer
        
        # Load model
        print("Loading GPT-2 model...")
        model_loader = MultiModelLoader()
        model, tokenizer = model_loader.load_model('gpt2')
        
        # Test weight analyzer
        print("Testing weight analyzer...")
        weight_analyzer = WeightAnalyzer(model)
        sparsity_analysis = weight_analyzer.get_sparsity_analysis()
        
        print(f"✅ Weight analyzer test successful")
        print(f"   - Overall sparsity: {sparsity_analysis['overall_sparsity']:.3f}")
        print(f"   - Total parameters: {sparsity_analysis['total_parameters']:,}")
        
        # Test pruning candidates
        candidates = weight_analyzer.identify_pruning_candidates(threshold_percentile=10.0)
        print(f"   - Pruning candidates: {len(candidates)}")
        
        # Test pruning impact analyzer
        print("Testing pruning impact analyzer...")
        pruning_analyzer = PruningImpactAnalyzer(model, tokenizer)
        
        # Test with a simple input
        impact = pruning_analyzer.simulate_neuron_pruning(0, [0, 1], "Hello world")
        print(f"✅ Pruning impact analyzer test successful")
        print(f"   - Impact score: {impact.impact_score:.3f}")
        print(f"   - Mean change: {impact.mean_change:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct module test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧠 NeuronScope Pruning Analysis Test Suite")
    print("=" * 60)
    
    # Test direct module functionality first
    direct_success = test_direct_module()
    
    # Test API endpoints
    api_success = test_api_server()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Direct module tests: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"   API endpoint tests: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if direct_success and api_success:
        print("\n🎉 All tests passed! Pruning analysis is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1) 