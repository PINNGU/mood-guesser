"""
CRDT State Manager that combines all CRDTs for mood classification.
"""

import time
from typing import Dict, Any, Set
from .g_counter import GCounter
from .or_set import ORSet
from .lww_map import LWWMap


class CRDTStateManager:
    """
    Manager class that combines all CRDTs for distributed mood classification.
    
    Manages mood counters, training rounds, sample counts, mood tags,
    and metadata using appropriate CRDT types.
    """
    
    def __init__(self, node_id: str):
        """
        Initialize the CRDT State Manager.
        
        Args:
            node_id: Unique identifier for the node
        """
        self.node_id = node_id
        
        # Dict of mood_name -> GCounter for counting moods
        self.mood_counters: Dict[str, GCounter] = {}
        
        # Counter for training rounds
        self.training_rounds = GCounter(node_id)
        
        # Counter for total samples processed
        self.total_samples = GCounter(node_id)
        
        # Set of known mood tags
        self.mood_tags = ORSet(node_id)
        
        # Metadata map for model info
        self.metadata = LWWMap(node_id)
        
        # Initialize metadata
        self.metadata.set('model_version', 0)
        self.metadata.set('last_update', 0)
    
    def increment_mood_count(self, mood: str, amount: int = 1) -> None:
        """
        Increment the counter for a specific mood.
        
        Args:
            mood: Name of the mood
            amount: Amount to increment by (default: 1)
        """
        # Create counter for mood if it doesn't exist
        if mood not in self.mood_counters:
            self.mood_counters[mood] = GCounter(self.node_id)
        
        self.mood_counters[mood].increment(amount)
        
        # Add mood to known tags
        if not self.mood_tags.contains(mood):
            self.mood_tags.add(mood)
    
    def get_mood_count(self, mood: str) -> int:
        """
        Get the count for a specific mood.
        
        Args:
            mood: Name of the mood
            
        Returns:
            Count for the mood (0 if doesn't exist)
        """
        if mood in self.mood_counters:
            return self.mood_counters[mood].value()
        return 0
    
    def increment_training_rounds(self) -> None:
        """Increment the training rounds counter by 1."""
        self.training_rounds.increment(1)
    
    def get_training_rounds(self) -> int:
        """
        Get the total number of training rounds.
        
        Returns:
            Total training rounds across all nodes
        """
        return self.training_rounds.value()
    
    def increment_total_samples(self, amount: int = 1) -> None:
        """
        Increment the total samples counter.
        
        Args:
            amount: Number of samples to add
        """
        self.total_samples.increment(amount)
    
    def get_total_samples(self) -> int:
        """
        Get the total number of samples processed.
        
        Returns:
            Total samples across all nodes
        """
        return self.total_samples.value()
    
    def update_model_version(self, version: int) -> None:
        """
        Update the model version in metadata.
        
        Args:
            version: New model version number
        """
        self.metadata.set('model_version', version)
        self.metadata.set('last_update', time.time())
    
    def get_model_version(self) -> int:
        """
        Get the current model version.
        
        Returns:
            Model version number (default 0)
        """
        return self.metadata.get('model_version', 0)
    
    def merge(self, other: 'CRDTStateManager') -> 'CRDTStateManager':
        """
        Merge this state manager with another.
        
        Merges all CRDTs and updates self with merged results.
        
        Args:
            other: Another CRDTStateManager instance
            
        Returns:
            Self with merged state
        """
        # Merge mood counters
        all_moods = set(self.mood_counters.keys()) | set(other.mood_counters.keys())
        for mood in all_moods:
            if mood in self.mood_counters and mood in other.mood_counters:
                self.mood_counters[mood] = self.mood_counters[mood].merge(
                    other.mood_counters[mood]
                )
            elif mood in other.mood_counters:
                # Create new counter with other's data
                self.mood_counters[mood] = GCounter(
                    self.node_id,
                    other.mood_counters[mood].counts.copy()
                )
        
        # Merge training rounds
        self.training_rounds = self.training_rounds.merge(other.training_rounds)
        
        # Merge total samples
        self.total_samples = self.total_samples.merge(other.total_samples)
        
        # Merge mood tags
        self.mood_tags = self.mood_tags.merge(other.mood_tags)
        
        # Merge metadata
        self.metadata = self.metadata.merge(other.metadata)
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize all CRDTs to a dictionary.
        
        Returns:
            Dictionary representation of all state
        """
        return {
            'type': 'CRDTStateManager',
            'node_id': self.node_id,
            'mood_counters': {
                mood: counter.to_dict()
                for mood, counter in self.mood_counters.items()
            },
            'training_rounds': self.training_rounds.to_dict(),
            'total_samples': self.total_samples.to_dict(),
            'mood_tags': self.mood_tags.to_dict(),
            'metadata': self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTStateManager':
        """
        Create a CRDTStateManager from a dictionary representation.
        
        Args:
            data: Dictionary containing serialized state
            
        Returns:
            A new CRDTStateManager instance
        """
        manager = cls(data['node_id'])
        
        # Restore mood counters
        manager.mood_counters = {
            mood: GCounter.from_dict(counter_data)
            for mood, counter_data in data.get('mood_counters', {}).items()
        }
        
        # Restore training rounds
        if 'training_rounds' in data:
            manager.training_rounds = GCounter.from_dict(data['training_rounds'])
        
        # Restore total samples
        if 'total_samples' in data:
            manager.total_samples = GCounter.from_dict(data['total_samples'])
        
        # Restore mood tags
        if 'mood_tags' in data:
            manager.mood_tags = ORSet.from_dict(data['mood_tags'])
        
        # Restore metadata
        if 'metadata' in data:
            manager.metadata = LWWMap.from_dict(data['metadata'])
        
        return manager
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a human-readable summary of the state.
        
        Returns:
            Dictionary with summary information
        """
        mood_counts = {
            mood: self.get_mood_count(mood)
            for mood in self.mood_counters.keys()
        }
        
        return {
            'node_id': self.node_id,
            'training_rounds': self.get_training_rounds(),
            'total_samples': self.get_total_samples(),
            'model_version': self.get_model_version(),
            'mood_counts': mood_counts,
            'known_moods': list(self.mood_tags.values())
        }
    
    def __repr__(self) -> str:
        return (
            f"CRDTStateManager(node_id={self.node_id}, "
            f"training_rounds={self.get_training_rounds()}, "
            f"model_version={self.get_model_version()}, "
            f"moods={list(self.mood_counters.keys())})"
        )
