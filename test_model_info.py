#!/usr/bin/env python3
"""
Test script for the Model Information Service

This script tests the comprehensive model information functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.backend.models.model_info_service import model_info_service

def test_model_info_service():
    """Test the model information service functionality."""
    
    print("🧠 Testing Model Information Service")
    print("=" * 50)
    
    # Test getting all models info
    print("\n1. Available Models:")
    all_models = model_info_service.get_all_models_info()
    for model_name, info in all_models.items():
        basic_info = info['basic_info']
        print(f"   • {model_name}: {basic_info['description']}")
        print(f"     Parameters: {basic_info['parameters']:,}")
        print(f"     Layers: {basic_info['layers']}")
        print(f"     Hidden Size: {basic_info['hidden_size']}")
        print()
    
    # Test comprehensive info for GPT-2
    print("2. Comprehensive GPT-2 Information:")
    gpt2_info = model_info_service.get_comprehensive_model_info('gpt2')
    
    print(f"   Basic Info:")
    basic = gpt2_info['basic_info']
    print(f"     • Model: {basic['description']}")
    print(f"     • Parameters: {basic['parameters']:,}")
    print(f"     • Layers: {basic['layers']}")
    print(f"     • Hidden Size: {basic['hidden_size']}")
    print(f"     • Attention Heads: {basic['num_attention_heads']}")
    
    print(f"\n   Architecture:")
    arch = gpt2_info['architecture']
    print(f"     • Type: {arch['overview']['model_type']}")
    print(f"     • Architecture: {arch['overview']['architecture']}")
    print(f"     • Activation: {arch['overview']['activation_function']}")
    
    print(f"\n   Parameter Distribution:")
    params = arch['parameter_distribution']
    for key, value in params.items():
        if key != 'total':
            print(f"     • {key.title()}: {value['parameters']:,} ({value['percentage']}%)")
    
    print(f"\n   Training Details:")
    training = gpt2_info['training_details']
    print(f"     • Dataset: {training['dataset']['name']} ({training['dataset']['size']})")
    print(f"     • Objective: {training['training_objective']['objective']}")
    print(f"     • Optimizer: {training['optimization']['optimizer']}")
    print(f"     • Learning Rate: {training['optimization']['learning_rate']}")
    
    print(f"\n   Performance:")
    perf = gpt2_info['performance_metrics']
    print(f"     • Memory Usage: {perf['model_size']['memory_footprint']}")
    print(f"     • Download Size: {perf['model_size']['download_size']}")
    print(f"     • CPU Speed: {perf['inference_speed']['cpu_speed']}")
    
    print(f"\n   Usage Guidelines:")
    usage = gpt2_info['usage_guidelines']
    print(f"     • Recommended Use Cases: {len(usage['recommended_use_cases'])} items")
    print(f"     • Limitations: {len(usage['limitations'])} items")
    print(f"     • Best Practices: {len(usage['best_practices'])} items")
    
    print(f"\n   Activation Equations:")
    equations = gpt2_info['activation_equations']
    print(f"     • Attention: {len(equations['attention_mechanism'])} equations")
    print(f"     • Feed-Forward: {len(equations['feed_forward'])} equations")
    print(f"     • Layer Norm: {len(equations['layer_norm'])} equations")
    
    print("\n✅ Model Information Service Test Completed Successfully!")
    
    return True

def test_error_handling():
    """Test error handling for invalid model names."""
    
    print("\n3. Testing Error Handling:")
    print("=" * 30)
    
    try:
        model_info_service.get_comprehensive_model_info('invalid-model')
        print("❌ Should have raised an error for invalid model")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    print("✅ Error Handling Test Passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_model_info_service() and test_error_handling()
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        sys.exit(1) 