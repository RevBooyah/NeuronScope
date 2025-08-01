"""
Data I/O utilities for NeuronScope

This module handles saving and loading of activation data, clustering results,
and other JSON-formatted data according to the structures defined in DATA_STRUCTURE.md.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DataIO:
    """Handles saving and loading of NeuronScope data."""
    
    def __init__(self, base_data_dir: str = "data"):
        """
        Initialize the DataIO handler.
        
        Args:
            base_data_dir: Base directory for data storage
        """
        self.base_data_dir = Path(base_data_dir)
        self.activations_dir = self.base_data_dir / "activations"
        self.clusters_dir = self.base_data_dir / "clusters"
        self.queries_dir = self.base_data_dir / "queries"
        
        # Create directories if they don't exist
        self.activations_dir.mkdir(parents=True, exist_ok=True)
        self.clusters_dir.mkdir(parents=True, exist_ok=True)
        self.queries_dir.mkdir(parents=True, exist_ok=True)
    
    def save_activations(self, activation_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save activation data to JSON file.
        
        Args:
            activation_data: Activation data dictionary
            filename: Optional filename. If None, generates from prompt.
            
        Returns:
            Path to saved file
        """
        if filename is None:
            # Generate filename from prompt
            prompt = activation_data["prompt"]
            # Clean prompt for filename
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit length
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_prompt}_{timestamp}.json"
        
        filepath = self.activations_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(activation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved activation data to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save activation data: {str(e)}")
            raise RuntimeError(f"Failed to save activation data: {str(e)}")
    
    def load_activations(self, filename: str) -> Dict[str, Any]:
        """
        Load activation data from JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Activation data dictionary
        """
        filepath = self.activations_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Activation file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                activation_data = json.load(f)
            
            logger.info(f"Loaded activation data from: {filepath}")
            return activation_data
            
        except Exception as e:
            logger.error(f"Failed to load activation data: {str(e)}")
            raise RuntimeError(f"Failed to load activation data: {str(e)}")
    
    def save_clusters(self, cluster_data: Dict[str, Any], filename: str) -> str:
        """
        Save clustering results to JSON file.
        
        Args:
            cluster_data: Clustering data dictionary
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        filepath = self.clusters_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cluster_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved cluster data to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save cluster data: {str(e)}")
            raise RuntimeError(f"Failed to save cluster data: {str(e)}")
    
    def load_clusters(self, filename: str) -> Dict[str, Any]:
        """
        Load clustering results from JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Clustering data dictionary
        """
        filepath = self.clusters_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Cluster file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)
            
            logger.info(f"Loaded cluster data from: {filepath}")
            return cluster_data
            
        except Exception as e:
            logger.error(f"Failed to load cluster data: {str(e)}")
            raise RuntimeError(f"Failed to load cluster data: {str(e)}")
    
    def save_queries(self, query_data: Dict[str, Any], filename: str) -> str:
        """
        Save reverse activation query results to JSON file.
        
        Args:
            query_data: Query data dictionary
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        filepath = self.queries_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved query data to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save query data: {str(e)}")
            raise RuntimeError(f"Failed to save query data: {str(e)}")
    
    def load_queries(self, filename: str) -> Dict[str, Any]:
        """
        Load reverse activation query results from JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Query data dictionary
        """
        filepath = self.queries_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Query file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                query_data = json.load(f)
            
            logger.info(f"Loaded query data from: {filepath}")
            return query_data
            
        except Exception as e:
            logger.error(f"Failed to load query data: {str(e)}")
            raise RuntimeError(f"Failed to load query data: {str(e)}")
    
    def list_activation_files(self) -> List[str]:
        """
        List all available activation files.
        
        Returns:
            List of activation filenames
        """
        files = list(self.activations_dir.glob("*.json"))
        return [f.name for f in files]
    
    def list_cluster_files(self) -> List[str]:
        """
        List all available cluster files.
        
        Returns:
            List of cluster filenames
        """
        files = list(self.clusters_dir.glob("*.json"))
        return [f.name for f in files]
    
    def list_query_files(self) -> List[str]:
        """
        List all available query files.
        
        Returns:
            List of query filenames
        """
        files = list(self.queries_dir.glob("*.json"))
        return [f.name for f in files]


def create_data_io(base_data_dir: str = "data") -> DataIO:
    """
    Factory function to create a DataIO instance.
    
    Args:
        base_data_dir: Base directory for data storage
        
    Returns:
        DataIO instance
    """
    return DataIO(base_data_dir=base_data_dir) 