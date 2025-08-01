#!/usr/bin/env python3
"""
Simple structure test for NeuronScope.
"""

import json
import sys
from pathlib import Path

def test_project_structure():
    """Test that the project structure is correct."""
    print("🧠 Testing NeuronScope Project Structure")
    print("=" * 50)
    
    # Check directories exist
    required_dirs = [
        "src/backend/models",
        "src/backend/activations", 
        "src/backend/clustering",
        "src/backend/queries",
        "src/backend/utils",
        "src/frontend",
        "data/activations",
        "data/clusters", 
        "data/queries",
        "scripts"
    ]
    
    print("📁 Checking directory structure...")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} (missing)")
    
    # Check key files exist
    required_files = [
        "requirements.txt",
        "samples.json",
        "README.md",
        "DATA_STRUCTURE.md",
        ".cursorrules"
    ]
    
    print("\n📄 Checking key files...")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
    
    # Test samples.json structure
    print("\n📋 Testing samples.json...")
    try:
        with open("samples.json", "r") as f:
            samples = json.load(f)
        print(f"  ✅ Loaded {len(samples)} sample prompts")
        for i, sample in enumerate(samples[:3]):  # Show first 3
            print(f"    {i+1}. {sample[:50]}...")
    except Exception as e:
        print(f"  ❌ Failed to load samples.json: {e}")
    
    # Test DATA_STRUCTURE.md
    print("\n📊 Testing DATA_STRUCTURE.md...")
    try:
        with open("DATA_STRUCTURE.md", "r") as f:
            content = f.read()
        if "Neuron Activation Output" in content:
            print("  ✅ DATA_STRUCTURE.md contains expected content")
        else:
            print("  ❌ DATA_STRUCTURE.md missing expected content")
    except Exception as e:
        print(f"  ❌ Failed to read DATA_STRUCTURE.md: {e}")
    
    print("\n✅ Structure test completed!")

if __name__ == "__main__":
    test_project_structure() 