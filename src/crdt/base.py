"""
Abstract base class for Conflict-free Replicated Data Types (CRDTs).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar

T = TypeVar('T', bound='CRDT')


class CRDT(ABC):
    """
    Abstract base class for all CRDTs.
    
    CRDTs are data structures that can be replicated across multiple nodes
    and merged without conflicts, ensuring eventual consistency.
    """
    
    def __init__(self, node_id: str):
        """
        Initialize the CRDT with a node identifier.
        
        Args:
            node_id: Unique identifier for the node owning this CRDT instance
        """
        self.node_id = node_id
    
    @abstractmethod
    def merge(self: T, other: T) -> T:
        """
        Merge this CRDT with another CRDT instance.
        
        The merge operation must be:
        - Commutative: merge(a, b) == merge(b, a)
        - Associative: merge(merge(a, b), c) == merge(a, merge(b, c))
        - Idempotent: merge(a, a) == a
        
        Args:
            other: Another CRDT instance of the same type
            
        Returns:
            A new CRDT instance representing the merged state
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the CRDT to a dictionary for storage or transmission.
        
        Returns:
            Dictionary representation of the CRDT state
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """
        Create a CRDT instance from a dictionary representation.
        
        Args:
            data: Dictionary containing the serialized CRDT state
            
        Returns:
            A new CRDT instance reconstructed from the dictionary
        """
        pass
