#!/usr/bin/env python3
"""
Script to install NeuronScope dependencies.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all required dependencies."""
    packages = [
        "pandas",
        "scikit-learn", 
        "matplotlib",
        "plotly",
        "click",
        "tqdm",
        "python-dotenv",
        "torch --index-url https://download.pytorch.org/whl/cpu",
        "transformers"
    ]
    
    print("🧠 Installing NeuronScope Dependencies")
    print("=" * 50)
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        print("Run 'python scripts/setup_models.py' to verify installation.")
    else:
        print("⚠️ Some packages failed to install. Check the errors above.")

if __name__ == "__main__":
    main() 