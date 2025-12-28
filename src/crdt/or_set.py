"""
Observed-Remove Set (OR-Set) CRDT implementation.
"""

import uuid
from typing import Dict, Any, Set
from .base import CRDT


class ORSet(CRDT):
    """
    Observed-Remove Set CRDT.
    
    An OR-Set allows both add and remove operations while maintaining
    CRDT properties. Each element is associated with unique tags that
    track its additions across nodes.
    """
    
    def __init__(self, node_id: str, elements: Dict[Any, Set[str]] = None):
        """
        Initialize the OR-Set.
        
        Args:
            node_id: Unique identifier for the node
            elements: Optional initial elements dictionary (element -> set of tags)
        """
        super().__init__(node_id)
        self.elements: Dict[Any, Set[str]] = {}
        if elements:
            for elem, tags in elements.items():
                self.elements[elem] = tags.copy()
    
    def _generate_tag(self) -> str:
        """
        Generate a unique tag for an element addition.
        
        Returns:
            Unique tag string in format "{node_id}:{uuid}"
        """
        return f"{self.node_id}:{uuid.uuid4()}"
    
    def add(self, element: Any) -> None:
        """
        Add an element to the set with a unique tag.
        
        Args:
            element: Element to add
        """
        tag = self._generate_tag()
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)
    
    def remove(self, element: Any) -> None:
        """
        Remove an element from the set.
        
        Args:
            element: Element to remove
        """
        if element in self.elements:
            del self.elements[element]
    
    def contains(self, element: Any) -> bool:
        """
        Check if an element exists in the set.
        
        Args:
            element: Element to check
            
        Returns:
            True if element exists with at least one tag
        """
        return element in self.elements and len(self.elements[element]) > 0
    
    def values(self) -> Set[Any]:
        """
        Get all elements in the set.
        
        Returns:
            Set of all element keys
        """
        return set(self.elements.keys())
    
    def merge(self, other: 'ORSet') -> 'ORSet':
        """
        Merge this OR-Set with another OR-Set.
        
        Unions all elements with their tags from both sets.
        
        Args:
            other: Another ORSet instance
            
        Returns:
            A new ORSet with merged elements and tags
        """
        merged = ORSet(self.node_id)
        
        # Get union of all elements
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        
        # Union tags for each element
        for element in all_elements:
            self_tags = self.elements.get(element, set())
            other_tags = other.elements.get(element, set())
            merged.elements[element] = self_tags | other_tags
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the OR-Set to a dictionary.
        
        Returns:
            Dictionary representation of the OR-Set
        """
        # Convert sets to lists for JSON serialization
        serialized_elements = {
            str(elem): list(tags) for elem, tags in self.elements.items()
        }
        return {
            'type': 'ORSet',
            'node_id': self.node_id,
            'elements': serialized_elements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ORSet':
        """
        Create an ORSet from a dictionary representation.
        
        Args:
            data: Dictionary containing serialized ORSet state
            
        Returns:
            A new ORSet instance
        """
        elements = {}
        for elem, tags in data.get('elements', {}).items():
            elements[elem] = set(tags)
        
        return cls(
            node_id=data['node_id'],
            elements=elements
        )
    
    def __repr__(self) -> str:
        values_list = list(self.values())
        preview = values_list[:5]
        more = f", ... +{len(values_list) - 5} more" if len(values_list) > 5 else ""
        return f"ORSet(size={len(values_list)}, elements={preview}{more})"
    
    def __len__(self) -> int:
        return len(self.elements)
