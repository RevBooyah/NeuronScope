"""
Neuron clustering for NeuronScope

This module implements clustering of neurons based on their activation patterns
to identify groups of neurons that respond similarly to inputs.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.gpt2_loader import GPT2ModelLoader
from activations.extractor import ActivationExtractor
from utils.data_io import DataIO

logger = logging.getLogger(__name__)

class NeuronClusterer:
    """Clusters neurons based on their activation patterns."""
    
    def __init__(self, model_loader: Optional[GPT2ModelLoader] = None, data_io: Optional[DataIO] = None):
        """
        Initialize the neuron clusterer.
        
        Args:
            model_loader: GPT-2 model loader instance
            data_io: Data I/O handler instance
        """
        self.model_loader = model_loader or GPT2ModelLoader()
        self.data_io = data_io or DataIO()
        self.extractor = None  # Will be set when model is loaded
        
        # Test prompts for clustering analysis
        self.test_prompts = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "hello", "world", "good", "bad", "big", "small", "new", "old", "high", "low",
            "time", "space", "place", "thing", "person", "people", "man", "woman", "child",
            "house", "car", "book", "computer", "phone", "food", "water", "air", "fire",
            "earth", "sun", "moon", "star", "tree", "flower", "animal", "bird", "fish",
            "dog", "cat", "horse", "cow", "sheep", "pig", "chicken", "duck", "goose"
        ]
    
    def extract_activation_matrix(self, layer_index: int, prompts: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract activation matrix for clustering analysis.
        
        Args:
            layer_index: Layer to analyze
            prompts: List of prompts to use. If None, uses default test prompts.
            
        Returns:
            Tuple of (activation_matrix, prompt_list)
        """
        if prompts is None:
            prompts = self.test_prompts
        
        logger.info(f"Extracting activation matrix for layer {layer_index} with {len(prompts)} prompts")
        
        # Load model and create extractor if not already done
        if self.extractor is None:
            model, tokenizer = self.model_loader.load_model('gpt2')
            self.extractor = ActivationExtractor(model, tokenizer)
        
        # Collect activations for each prompt
        activation_vectors = []
        successful_prompts = []
        
        for prompt in prompts:
            try:
                # Extract activations for this prompt
                activation_data = self.extractor.extract_activations(prompt)
                
                # Get activations for the specified layer
                layer_data = activation_data["layers"][layer_index]
                
                # Extract activation vector (average across tokens if multiple)
                layer_activations = []
                for neuron in layer_data["neurons"]:
                    # Average activation across tokens
                    avg_activation = np.mean(neuron["activations"])
                    layer_activations.append(float(avg_activation))
                
                activation_vectors.append(layer_activations)
                successful_prompts.append(prompt)
                
            except Exception as e:
                logger.warning(f"Failed to process prompt '{prompt}': {str(e)}")
                continue
        
        if not activation_vectors:
            raise RuntimeError("No successful activation extractions")
        
        # Convert to numpy array
        activation_matrix = np.array(activation_vectors)
        
        logger.info(f"Created activation matrix: {activation_matrix.shape}")
        return activation_matrix, successful_prompts
    
    def cluster_neurons(self, layer_index: int, n_clusters: int = 5, 
                       prompts: Optional[List[str]] = None, 
                       use_pca: bool = True, pca_components: int = 50) -> Dict[str, Any]:
        """
        Cluster neurons in a specific layer based on their activation patterns.
        
        Args:
            layer_index: Layer to cluster
            n_clusters: Number of clusters to create
            prompts: List of prompts to use for clustering
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components to use
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Clustering neurons in layer {layer_index} into {n_clusters} clusters")
        
        # Extract activation matrix
        activation_matrix, successful_prompts = self.extract_activation_matrix(layer_index, prompts)
        
        # Transpose to get neurons as rows, prompts as columns
        neuron_matrix = activation_matrix.T  # Shape: (n_neurons, n_prompts)
        
        # Standardize the data
        scaler = StandardScaler()
        neuron_matrix_scaled = scaler.fit_transform(neuron_matrix)
        
        # Apply PCA if requested
        if use_pca and pca_components < neuron_matrix_scaled.shape[1]:
            pca = PCA(n_components=pca_components)
            neuron_matrix_reduced = pca.fit_transform(neuron_matrix_scaled)
            explained_variance_ratio = pca.explained_variance_ratio_.tolist()
            logger.info(f"Applied PCA: {pca_components} components explain {sum(explained_variance_ratio):.2%} of variance")
        else:
            neuron_matrix_reduced = neuron_matrix_scaled
            explained_variance_ratio = None
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(neuron_matrix_reduced)
        cluster_centers = kmeans.cluster_centers_
        
        # Calculate cluster statistics
        clusters = []
        for cluster_id in range(n_clusters):
            # Find neurons in this cluster
            cluster_neuron_indices = np.where(cluster_labels == cluster_id)[0].tolist()
            
            # Calculate cluster statistics
            cluster_activations = neuron_matrix[cluster_neuron_indices]
            cluster_mean = np.mean(cluster_activations, axis=0).tolist()
            cluster_std = np.std(cluster_activations, axis=0).tolist()
            
            # Get cluster center (in original space if PCA was used)
            if use_pca and explained_variance_ratio is not None:
                # Transform cluster center back to original space
                cluster_center_original = pca.inverse_transform(cluster_centers[cluster_id])
                cluster_center_original = scaler.inverse_transform(cluster_center_original.reshape(1, -1)).flatten()
                centroid = cluster_center_original.tolist()
            else:
                centroid = cluster_centers[cluster_id].tolist()
            
            clusters.append({
                "cluster_id": cluster_id,
                "neuron_indices": cluster_neuron_indices,
                "size": len(cluster_neuron_indices),
                "centroid": centroid,
                "mean_activations": cluster_mean,
                "std_activations": cluster_std
            })
        
        # Create result structure
        result = {
            "layer": layer_index,
            "n_clusters": n_clusters,
            "n_neurons": len(cluster_labels),
            "n_prompts": len(successful_prompts),
            "prompts": successful_prompts,
            "clusters": clusters,
            "cluster_labels": cluster_labels.tolist(),
            "pca_info": {
                "used": use_pca,
                "n_components": pca_components if use_pca else None,
                "explained_variance_ratio": explained_variance_ratio
            },
            "clustering_algorithm": "KMeans"
        }
        
        logger.info(f"Clustering complete: {len(clusters)} clusters created")
        return result
    
    def save_clustering_result(self, clustering_result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save clustering result to JSON file.
        
        Args:
            clustering_result: Clustering result dictionary
            filename: Optional filename. If None, generates from clustering info.
            
        Returns:
            Path to saved file
        """
        if filename is None:
            # Generate filename from clustering info
            layer_idx = clustering_result["layer"]
            n_clusters = clustering_result["n_clusters"]
            filename = f"layer_{layer_idx}_{n_clusters}_clusters.json"
        
        return self.data_io.save_clusters(clustering_result, filename)
    
    def load_clustering_result(self, filename: str) -> Dict[str, Any]:
        """
        Load clustering result from JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Clustering result dictionary
        """
        return self.data_io.load_clusters(filename)
    
    def list_clustering_files(self) -> List[str]:
        """
        List all available clustering result files.
        
        Returns:
            List of clustering result filenames
        """
        return self.data_io.list_cluster_files()
    
    def analyze_cluster_characteristics(self, clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze characteristics of clusters to understand their behavior.
        
        Args:
            clustering_result: Result from cluster_neurons
            
        Returns:
            Dictionary with cluster analysis
        """
        clusters = clustering_result["clusters"]
        prompts = clustering_result["prompts"]
        
        analysis = {
            "layer": clustering_result["layer"],
            "cluster_analyses": []
        }
        
        for cluster in clusters:
            # Find most distinctive prompts for this cluster
            mean_activations = cluster["mean_activations"]
            std_activations = cluster["std_activations"]
            
            # Calculate z-scores for each prompt
            z_scores = []
            for i, (mean, std) in enumerate(zip(mean_activations, std_activations)):
                if std > 0:
                    z_score = abs(mean / std)
                else:
                    z_score = 0
                z_scores.append((i, z_score))
            
            # Sort by z-score to find most distinctive prompts
            z_scores.sort(key=lambda x: x[1], reverse=True)
            top_prompts = [(prompts[idx], z_score) for idx, z_score in z_scores[:5]]
            
            cluster_analysis = {
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "top_characteristic_prompts": top_prompts,
                "mean_activation_magnitude": np.mean(np.abs(mean_activations)),
                "activation_variability": np.mean(std_activations)
            }
            
            analysis["cluster_analyses"].append(cluster_analysis)
        
        return analysis 