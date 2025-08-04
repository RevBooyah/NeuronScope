"""
Flask API server for NeuronScope backend

This server provides REST API endpoints for the React frontend to interact
with the backend functionality (activation extraction, clustering, queries).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
from pathlib import Path

# Add the backend directory to the path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from models.gpt2_loader import MultiModelLoader
from activations.extractor import ActivationExtractor
from clustering.neuron_clustering import NeuronClusterer
from queries.reverse_queries import ReverseQueryEngine
from utils.data_io import DataIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize backend components (lazy loading)
data_io = DataIO()
model_loader = None  # Will be initialized when needed
extractor = None  # Will be initialized when needed
clusterer = None  # Will be initialized when needed
query_engine = None  # Will be initialized when needed
current_model = 'gpt2'  # Default model

def get_model_loader():
    """Get or create the model loader."""
    global model_loader
    if model_loader is None:
        model_loader = MultiModelLoader(use_quantization=True)  # Enable quantization for large models
    return model_loader

def get_clusterer():
    """Get or create the clusterer."""
    global clusterer
    if clusterer is None:
        clusterer = NeuronClusterer(get_model_loader(), data_io)
    return clusterer

def get_query_engine():
    """Get or create the query engine."""
    global query_engine
    if query_engine is None:
        query_engine = ReverseQueryEngine(get_model_loader(), data_io)
    return query_engine

def get_extractor():
    """Get or create the activation extractor."""
    global extractor
    if extractor is None:
        model, tokenizer = get_model_loader().load_model('gpt2')
        extractor = ActivationExtractor(model, tokenizer)
    return extractor

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'NeuronScope API server is running'
    })

@app.route('/api/activations/files', methods=['GET'])
def get_activation_files():
    """Get list of available activation files."""
    try:
        files = data_io.list_activation_files()
        return jsonify({
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        logger.error(f"Failed to get activation files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/activations/<filename>', methods=['GET'])
def get_activation_data(filename: str):
    """Get activation data for a specific file."""
    try:
        data = data_io.load_activations(filename)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Failed to load activation data for {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/activations/generate', methods=['POST'])
def generate_activation():
    """Generate new activation data for a prompt."""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        logger.info(f"Generating activation for prompt: {prompt}")
        
        # Get extractor and generate activation
        extractor = get_extractor()
        activation_data = extractor.extract_activations(prompt)
        
        # Save the activation data
        filename = data_io.save_activations(activation_data)
        
        return jsonify({
            'data': activation_data,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Failed to generate activation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters/files', methods=['GET'])
def get_cluster_files():
    """Get list of available cluster files."""
    try:
        files = data_io.list_cluster_files()
        return jsonify({
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        logger.error(f"Failed to get cluster files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters/<filename>', methods=['GET'])
def get_cluster_data(filename: str):
    """Get cluster data for a specific file."""
    try:
        data = data_io.load_clusters(filename)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Failed to load cluster data for {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters/generate', methods=['POST'])
def generate_clusters():
    """Generate new cluster data for a layer."""
    try:
        data = request.get_json()
        layer_index = data.get('layer_index', 0)
        n_clusters = data.get('n_clusters', 5)
        
        logger.info(f"Generating clusters for layer {layer_index} with {n_clusters} clusters")
        
        # Generate clustering
        clusterer = get_clusterer()
        clustering_result = clusterer.cluster_neurons(
            layer_index=layer_index,
            n_clusters=n_clusters
        )
        
        # Save the clustering result
        filename = clusterer.save_clustering_result(clustering_result)
        
        return jsonify({
            'data': clustering_result,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Failed to generate clusters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/queries/files', methods=['GET'])
def get_query_files():
    """Get list of available query files."""
    try:
        files = data_io.list_query_files()
        return jsonify({
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        logger.error(f"Failed to get query files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/queries/<filename>', methods=['GET'])
def get_query_data(filename: str):
    """Get query data for a specific file."""
    try:
        data = data_io.load_queries(filename)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Failed to load query data for {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/queries/neuron', methods=['POST'])
def query_neuron():
    """Query a specific neuron for activating tokens."""
    try:
        data = request.get_json()
        neuron_index = data.get('neuron_index')
        layer_index = data.get('layer_index', 0)
        top_k = data.get('top_k', 10)
        
        if neuron_index is None:
            return jsonify({'error': 'neuron_index is required'}), 400
        
        logger.info(f"Querying neuron {neuron_index} in layer {layer_index}")
        
        # Perform query
        query_engine = get_query_engine()
        query_result = query_engine.query_neuron_activations(
            neuron_index=neuron_index,
            layer_index=layer_index,
            top_k=top_k
        )
        
        # Save the query result
        filename = query_engine.save_query_result(query_result)
        
        return jsonify({
            'data': query_result,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Failed to query neuron: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/queries/cluster', methods=['POST'])
def query_cluster():
    """Query a cluster for activating tokens."""
    try:
        data = request.get_json()
        cluster_indices = data.get('cluster_indices')
        layer_index = data.get('layer_index', 0)
        top_k = data.get('top_k', 10)
        
        if not cluster_indices:
            return jsonify({'error': 'cluster_indices is required'}), 400
        
        logger.info(f"Querying cluster with {len(cluster_indices)} neurons in layer {layer_index}")
        
        # Perform query
        query_engine = get_query_engine()
        query_result = query_engine.query_cluster_activations(
            cluster_indices=cluster_indices,
            layer_index=layer_index,
            top_k=top_k
        )
        
        # Save the query result
        filename = query_engine.save_query_result(query_result)
        
        return jsonify({
            'data': query_result,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Failed to query cluster: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/samples', methods=['GET'])
def get_sample_prompts():
    """Get sample prompts."""
    try:
        # Load from samples.json if it exists
        samples_file = Path('samples.json')
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                samples = json.load(f)
        else:
            # Fallback samples
            samples = [
                "Hello world",
                "What is the capital of France?",
                "Translate this sentence to Spanish.",
                "The cat sat on the mat.",
                "If it rains tomorrow, we'll cancel the picnic."
            ]
        
        return jsonify(samples)
        
    except Exception as e:
        logger.error(f"Failed to get sample prompts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/samples', methods=['POST'])
def add_sample_prompt():
    """Add a new sample prompt."""
    try:
        data = request.get_json()
        new_prompt = data.get('prompt', '').strip()
        
        if not new_prompt:
            return jsonify({'error': 'prompt is required'}), 400
        
        # Load existing samples
        samples_file = Path('samples.json')
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                samples = json.load(f)
        else:
            samples = []
        
        # Check if prompt already exists
        if new_prompt in samples:
            return jsonify({'error': 'Prompt already exists in samples'}), 400
        
        # Add new prompt
        samples.append(new_prompt)
        
        # Save updated samples
        with open(samples_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Added new sample prompt: {new_prompt}")
        
        return jsonify({
            'message': 'Prompt added successfully',
            'prompt': new_prompt,
            'total_samples': len(samples)
        })
        
    except Exception as e:
        logger.error(f"Failed to add sample prompt: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models."""
    try:
        model_loader = get_model_loader()
        models = model_loader.get_available_models()
        recommended = model_loader.get_recommended_models()
        
        return jsonify({
            'models': models,
            'recommended': recommended,
            'current_model': current_model,
            'memory_usage': model_loader.get_memory_usage()
        })
        
    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_name>', methods=['GET'])
def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        model_loader = get_model_loader()
        info = model_loader.get_model_info(model_name)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model."""
    global current_model, extractor, clusterer, query_engine
    
    try:
        data = request.get_json()
        new_model = data.get('model_name', '').strip()
        
        if not new_model:
            return jsonify({'error': 'model_name is required'}), 400
        
        model_loader = get_model_loader()
        
        # Check if model is supported
        if new_model not in model_loader.get_available_models():
            available = list(model_loader.get_available_models().keys())
            return jsonify({'error': f'Model not supported. Available: {available}'}), 400
        
        logger.info(f"Switching from {current_model} to {new_model}")
        
        # Load the new model
        model, tokenizer = model_loader.load_model(new_model)
        
        # Update current model
        current_model = new_model
        
        # Reset components that depend on the model
        extractor = None
        clusterer = None
        query_engine = None
        
        # Get model info
        model_info = model_loader.get_model_info(new_model)
        
        return jsonify({
            'message': f'Successfully switched to {new_model}',
            'current_model': current_model,
            'model_info': model_info,
            'memory_usage': model_loader.get_memory_usage()
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to switch model: {error_msg}")
        
        # Provide more specific error messages
        if "authentication" in error_msg.lower() or "hf_token" in error_msg.lower():
            return jsonify({
                'error': f'Authentication required for {new_model}. Please set HF_TOKEN environment variable or login with huggingface-cli.',
                'details': error_msg,
                'requires_auth': True
            }), 401
        elif "gated repo" in error_msg.lower() or "access" in error_msg.lower():
            return jsonify({
                'error': f'Access denied for {new_model}. This model requires special access permissions.',
                'details': error_msg,
                'requires_auth': True
            }), 403
        elif "trust_remote_code" in error_msg.lower():
            return jsonify({
                'error': f'Failed to load {new_model}. This model requires custom code execution.',
                'details': error_msg
            }), 500
        elif "bitsandbytes" in error_msg.lower():
            return jsonify({
                'error': f'Failed to load {new_model}. Quantization library not properly installed.',
                'details': error_msg
            }), 500
        else:
            return jsonify({
                'error': f'Failed to switch to {new_model}',
                'details': error_msg
            }), 500

@app.route('/api/models/memory', methods=['GET'])
def get_memory_usage():
    """Get current memory usage information."""
    try:
        model_loader = get_model_loader()
        memory_info = model_loader.get_memory_usage()
        
        return jsonify(memory_info)
        
    except Exception as e:
        logger.error(f"Failed to get memory usage: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Serve static files from the data directory
@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """Serve data files for the frontend."""
    return send_from_directory('data', filename)

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🧠 Starting NeuronScope API server on port {port}")
    print(f"📁 Data directory: {data_io.base_data_dir}")
    print(f"🌐 API endpoints available at: http://localhost:{port}/api/")
    
    app.run(host='0.0.0.0', port=port, debug=True) 