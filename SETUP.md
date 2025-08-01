# NeuronScope Setup Guide

This guide will help you set up NeuronScope for local development and research.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- At least 2GB of free disk space (for models and data)

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd nscope

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip and install core tools
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install numpy pandas scikit-learn matplotlib plotly click tqdm python-dotenv

# Install PyTorch (CPU version - faster download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Transformers
pip install transformers

# Or install all at once from requirements.txt
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Run the setup verification script
python scripts/setup_models.py
```

You should see:
```
✅ PyTorch X.X.X available
✅ Transformers X.X.X available  
✅ NumPy X.X.X available
✅ All dependencies available!
```

### 4. Test Activation Extraction

```bash
# Test with a simple prompt
python test_activation.py
```

## Project Structure

```
nscope/
├── src/
│   ├── backend/
│   │   ├── models/          # GPT-2 model loading
│   │   ├── activations/     # Activation extraction
│   │   ├── clustering/      # Neuron clustering
│   │   ├── queries/         # Reverse activation queries
│   │   └── utils/           # Utility functions
│   └── frontend/            # React frontend (coming soon)
├── data/
│   ├── activations/         # Generated activation JSON
│   ├── clusters/            # Generated cluster JSON
│   └── queries/             # Generated query JSON
├── scripts/                 # CLI tools
├── samples.json             # Sample prompts for testing
└── requirements.txt         # Python dependencies
```

## Available Models

NeuronScope supports three GPT-2 model variants:

| Model | Parameters | Layers | Hidden Size | Download Size | Recommended |
|-------|------------|--------|-------------|---------------|-------------|
| gpt2 | 124M | 12 | 768 | ~500MB | ✅ Yes |
| gpt2-medium | 355M | 24 | 1024 | ~1.4GB | ✅ Yes |
| gpt2-large | 774M | 36 | 1280 | ~3GB | ⚠️ Large |

## Usage Examples

### Extract Activations

```bash
# Process all sample prompts
python scripts/extract_activations.py

# Process specific prompts (coming soon)
python scripts/extract_activations.py --prompt "Hello world"
```

### View Results

Generated activation files are saved in `data/activations/` with the format:
- `{prompt_name}_{timestamp}.json`

Each file contains:
- Original prompt and tokens
- Neuron activations for each layer
- Activation statistics

## Troubleshooting

### Common Issues

**1. PyTorch Installation Fails**
```bash
# Try CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or try conda
conda install pytorch cpuonly -c pytorch
```

**2. Out of Memory Errors**
- Use smaller models (gpt2 instead of gpt2-large)
- Process shorter prompts
- Close other applications

**3. Slow Performance**
- Ensure you're using the virtual environment
- Consider GPU installation for faster processing
- Use smaller models for testing

### GPU Support (Optional)

For faster processing, install GPU-enabled PyTorch:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Development Workflow

1. **Backend Development**: Python modules in `src/backend/`
2. **Data Generation**: CLI scripts in `scripts/`
3. **Frontend Development**: React app in `src/frontend/` (coming soon)
4. **Testing**: Use sample prompts in `samples.json`

## Next Steps

After successful setup:

1. ✅ Extract activations from sample prompts
2. 🔄 Implement clustering algorithms
3. 🔄 Build React frontend for visualization
4. 🔄 Add reverse activation queries
5. 🔄 Implement neuron drift analysis

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Check the project's TODO.md for known issues
4. Review the DATA_STRUCTURE.md for expected data formats

## Contributing

See CONTRIBUTING.md for development guidelines and contribution instructions. 