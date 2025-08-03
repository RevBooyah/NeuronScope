#!/usr/bin/env python3
"""
Test script for NeuronScope API Server

This script tests the Flask API server endpoints to ensure they're working correctly.
"""

import sys
import requests
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_api_server():
    """Test the API server endpoints."""
    print("🧠 Testing NeuronScope API Server")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test health check
        print("🔍 Testing health check...")
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   - Status: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test getting activation files
        print("\n🔍 Testing activation files endpoint...")
        response = requests.get(f"{base_url}/api/activations/files")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Activation files endpoint working")
            print(f"   - Files found: {data['count']}")
            if data['files']:
                print(f"   - Sample file: {data['files'][0]}")
        else:
            print(f"❌ Activation files endpoint failed: {response.status_code}")
        
        # Test getting sample prompts
        print("\n🔍 Testing sample prompts endpoint...")
        response = requests.get(f"{base_url}/api/samples")
        if response.status_code == 200:
            prompts = response.json()
            print(f"✅ Sample prompts endpoint working")
            print(f"   - Prompts found: {len(prompts)}")
            if prompts:
                print(f"   - Sample prompt: {prompts[0]}")
        else:
            print(f"❌ Sample prompts endpoint failed: {response.status_code}")
        
        # Test getting cluster files
        print("\n🔍 Testing cluster files endpoint...")
        response = requests.get(f"{base_url}/api/clusters/files")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cluster files endpoint working")
            print(f"   - Files found: {data['count']}")
        else:
            print(f"❌ Cluster files endpoint failed: {response.status_code}")
        
        # Test getting query files
        print("\n🔍 Testing query files endpoint...")
        response = requests.get(f"{base_url}/api/queries/files")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query files endpoint working")
            print(f"   - Files found: {data['count']}")
        else:
            print(f"❌ Query files endpoint failed: {response.status_code}")
        
        print("\n🎉 API server tests completed!")
        print("\n📝 Next steps:")
        print("1. Start the React frontend")
        print("2. Test the full integration")
        print("3. Add more API endpoints as needed")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("   - Make sure the server is running on http://localhost:5000")
        print("   - Run: python src/backend/api_server.py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_api_server()
    sys.exit(0 if success else 1) 