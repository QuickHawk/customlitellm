import pickle
from typing import Dict, Any, List, Optional
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from .base_template_router import BaseTemplateRouter

class TemplateRouter(BaseTemplateRouter):
    """
    A template-based router for LiteLLM that uses drain3 for prompt clustering.
    Designed to replace SemanticRouter with similar function signatures.
    """
    
    def __init__(
        self, 
        persistence_file: str = "prompt_templates.bin",
        metadata_file: str = "cluster_metadata.pkl",
        persistence_handler: Optional[FilePersistence] = None,
    ):
        """
        Initialize the TemplateRouter.
        
        Args:
            persistence_file: Path to the persistence file (default: "prompt_templates.bin")
            metadata_file: Path to the metadata persistence file (default: "cluster_metadata.pkl")
            persistence_handler: Optional custom persistence handler
        """
        if persistence_handler is None:
            persistence_handler = FilePersistence(persistence_file)
        
        self.persistence_handler = persistence_handler
        self.metadata_file = metadata_file
        self.template_miner = TemplateMiner(
            persistence_handler=persistence_handler,
        )
        self._cluster_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Load metadata from file if it exists
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cluster metadata from the persistence file."""
        try:
            with open(self.metadata_file, 'rb') as f:
                self._cluster_metadata = pickle.load(f)
        except FileNotFoundError:
            # File doesn't exist yet, start with empty metadata
            self._cluster_metadata = {}
        except Exception as e:
            print(f"Warning: Could not load metadata from {self.metadata_file}: {e}")
            self._cluster_metadata = {}
    
    def _save_metadata(self):
        """Save cluster metadata to the persistence file."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self._cluster_metadata, f)
        except Exception as e:
            print(f"Error: Could not save metadata to {self.metadata_file}: {e}")
    
    def __call__(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Route a prompt to its matching template cluster.
        
        Args:
            text: The input prompt/message to route
            
        Returns:
            Dictionary with routing information if matched, None otherwise
        """
        result = self.match(text)
        return result
    
    def add_log_message(self, message: str) -> Dict[str, Any]:
        """
        Add a message/prompt to the template miner.
        
        Args:
            message: The prompt or log message to add
            
        Returns:
            Dictionary containing clustering result with keys:
            - 'cluster_id': ID of the cluster
            - 'template_mined': The extracted template
            - 'change_type': Type of change made to clusters
        """
        result = self.template_miner.add_log_message(message.strip())
        
        return {
            'cluster_id': result['cluster_id'],
            'template_mined': result['template_mined'],
            'change_type': result['change_type']
        }
    
    def match(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Match a message to an existing template cluster.
        
        Args:
            message: The message/prompt to match
            
        Returns:
            Dictionary with match information if found, None otherwise.
            Contains:
            - 'cluster_id': ID of the matched cluster
            - 'template': The template pattern
            - 'size': Number of messages in the cluster
            - 'name': Cluster name if metadata exists
        """
        matched = self.template_miner.match(message.strip())
        
        if matched:
            cluster_id = matched.cluster_id
            return {
                'cluster_id': cluster_id,
                'template': matched.get_template(),
                'size': matched.size,
                'name': self._cluster_metadata.get(cluster_id, {}).get('name', f'cluster_{cluster_id}'),
                'target_model': self._cluster_metadata.get(cluster_id, {}).get('target_model', None)
            }
        
        return None
    
    def get_all_templates(self) -> List[Dict[str, Any]]:
        """
        Get all existing template clusters.
        
        Returns:
            List of dictionaries containing cluster information
        """
        templates = []
        for cluster in self.template_miner.drain.clusters:
            cluster_id = cluster.cluster_id
            templates.append({
                'cluster_id': cluster_id,
                'template': cluster.get_template(),
                'size': cluster.size,
                'name': self._cluster_metadata.get(cluster_id, {}).get('name', f'cluster_{cluster_id}')
            })
        return templates
    
    def set_cluster_metadata(self, cluster_id: int, name: str = "", **kwargs):
        """
        Set metadata for a specific cluster (useful for routing decisions later).
        Automatically saves metadata to file after updating.
        
        Args:
            cluster_id: The ID of the cluster to update
            name: Human-readable name for the cluster
            **kwargs: Additional metadata to store
        """
        if cluster_id not in self._cluster_metadata:
            self._cluster_metadata[cluster_id] = {}
        
        if name:
            self._cluster_metadata[cluster_id]['name'] = name
        
        self._cluster_metadata[cluster_id].update(kwargs)
        
        # Persist metadata after updating
        self._save_metadata()
    
    def print_summary(self):
        """Print a summary of all templates."""
        print("=" * 60)
        print("ALL TEMPLATES:")
        print("=" * 60)
        for cluster in self.template_miner.drain.clusters:
            print(f"Cluster ID: {cluster.cluster_id}")
            print(f"Template: {cluster.get_template()}")
            print(f"Size: {cluster.size}")
            metadata = self._cluster_metadata.get(cluster.cluster_id, {})
            if metadata:
                print(f"Metadata: {metadata}")
            print("-" * 60)
