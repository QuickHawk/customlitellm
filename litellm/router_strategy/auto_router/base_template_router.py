from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseTemplateRouter(ABC):
    """
    Abstract base class for a template-based router.
    Defines the interface for different template routing strategies.
    """

    @abstractmethod
    def __call__(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Route a prompt to its matching template cluster.
        
        Args:
            text: The input prompt/message to route
            
        Returns:
            Dictionary with routing information if matched, None otherwise
        """
        pass

    @abstractmethod
    def add_log_message(self, message: str) -> Dict[str, Any]:
        """
        Add a message/prompt to the template miner.
        
        Args:
            message: The prompt or log message to add
            
        Returns:
            Dictionary containing clustering result.
        """
        pass

    @abstractmethod
    def match(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Match a message to an existing template cluster.
        
        Args:
            message: The message/prompt to match
            
        Returns:
            Dictionary with match information if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_all_templates(self) -> List[Dict[str, Any]]:
        """
        Get all existing template clusters.
        
        Returns:
            List of dictionaries containing cluster information
        """
        pass

    @abstractmethod
    def set_cluster_metadata(self, cluster_id: int, name: str = "", **kwargs):
        """
        Set metadata for a specific cluster.
        
        Args:
            cluster_id: The ID of the cluster to update
            name: Human-readable name for the cluster
            **kwargs: Additional metadata to store
        """
        pass

    @abstractmethod
    def print_summary(self):
        """Print a summary of all templates."""
        pass
