#!/usr/bin/env python3
"""
Test script to verify current package installation status.
"""

def test_imports():
    """Test which packages can be imported."""
    print("🧠 Testing NeuronScope Package Installation")
    print("=" * 50)
    
    packages = {
        'numpy': 'NumPy - Numerical computing',
        'pandas': 'Pandas - Data manipulation',
        'sklearn': 'Scikit-learn - Machine learning',
        'torch': 'PyTorch - Deep learning framework',
        'transformers': 'Transformers - Hugging Face models',
        'matplotlib': 'Matplotlib - Plotting',
        'plotly': 'Plotly - Interactive plots',
        'click': 'Click - CLI framework',
        'tqdm': 'TQDM - Progress bars'
    }
    
    working_packages = []
    missing_packages = []
    
    for package, description in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {package} ({version}) - {description}")
            working_packages.append(package)
            
        except ImportError:
            print(f"❌ {package} - {description}")
            missing_packages.append(package)
    
    print(f"\n📊 Summary:")
    print(f"✅ Working packages: {len(working_packages)}/{len(packages)}")
    print(f"❌ Missing packages: {len(missing_packages)}")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("\n💡 To install missing packages:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("   pip install transformers matplotlib plotly click tqdm")
    
    return len(working_packages) >= 3  # At least numpy, pandas, sklearn

def test_basic_functionality():
    """Test basic functionality with available packages."""
    print(f"\n🔧 Testing Basic Functionality")
    print("=" * 30)
    
    try:
        import numpy as np
        import pandas as pd
        
        # Test NumPy
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✅ NumPy array creation: {arr}")
        
        # Test Pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✅ Pandas DataFrame creation: {df.shape}")
        
        # Test scikit-learn
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2)
            print("✅ Scikit-learn KMeans import successful")
        except ImportError:
            print("❌ Scikit-learn clustering not available")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    imports_ok = test_imports()
    if imports_ok:
        test_basic_functionality()
    
    print(f"\n📝 Next steps:")
    if imports_ok:
        print("1. Install remaining packages if needed")
        print("2. Test activation extraction")
        print("3. Start building visualizations")
    else:
        print("1. Fix package installation issues")
        print("2. Ensure virtual environment is activated")
        print("3. Check Python version compatibility") 