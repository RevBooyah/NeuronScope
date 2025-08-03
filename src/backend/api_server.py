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

from models.gpt2_loader import GPT2ModelLoader
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

def get_model_loader():
    """Get or create the model loader."""
    global model_loader
    if model_loader is None:
        model_loader = GPT2ModelLoader()
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