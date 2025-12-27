"""
Grow-only Counter (G-Counter) CRDT implementation.
"""

from typing import Dict, Any
from .base import CRDT


class GCounter(CRDT):
    """
    Grow-only Counter CRDT.
    
    A G-Counter can only be incremented, never decremented.
    Each node maintains its own count, and the total value
    is the sum of all node counts.
    """
    
    def __init__(self, node_id: str, counts: Dict[str, int] = None):
        """
        Initialize the G-Counter.
        
        Args:
            node_id: Unique identifier for the node
            counts: Optional initial counts dictionary
        """
        super().__init__(node_id)
        self.counts: Dict[str, int] = counts.copy() if counts else {}
    
    def increment(self, amount: int = 1) -> None:
        """
        Increment the counter for this node.
        
        Args:
            amount: Amount to increment by (default: 1)
        """
        if amount < 0:
            raise ValueError("G-Counter can only be incremented with positive values")
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount
    
    def value(self) -> int:
        """
        Get the current total value of the counter.
        
        Returns:
            Sum of all node counts
        """
        return sum(self.counts.values())
    
    def merge(self, other: 'GCounter') -> 'GCounter':
        """
        Merge this G-Counter with another G-Counter.
        
        Takes the maximum count for each node from both counters.
        
        Args:
            other: Another GCounter instance
            
        Returns:
            A new GCounter with merged counts
        """
        merged = GCounter(self.node_id)
        
        # Get union of all node IDs
        all_nodes = set(self.counts.keys()) | set(other.counts.keys())
        
        # Take max count for each node
        for node in all_nodes:
            merged.counts[node] = max(
                self.counts.get(node, 0),
                other.counts.get(node, 0)
            )
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the G-Counter to a dictionary.
        
        Returns:
            Dictionary representation of the G-Counter
        """
        return {
            'type': 'GCounter',
            'node_id': self.node_id,
            'counts': self.counts.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GCounter':
        """
        Create a GCounter from a dictionary representation.
        
        Args:
            data: Dictionary containing serialized GCounter state
            
        Returns:
            A new GCounter instance
        """
        return cls(
            node_id=data['node_id'],
            counts=data.get('counts', {})
        )
    
    def __repr__(self) -> str:
        return f"GCounter(value={self.value()}, counts={self.counts})"
