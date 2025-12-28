"""
Last-Write-Wins Map (LWW-Map) CRDT implementation.
"""

import time
from typing import Dict, Any, Set, Tuple, Optional
from .base import CRDT


class LWWMap(CRDT):
    """
    Last-Write-Wins Map CRDT.
    
    An LWW-Map resolves conflicts by keeping the entry with the
    latest timestamp. Each value is stored alongside its timestamp.
    """
    
    def __init__(self, node_id: str, data: Dict[str, Tuple[Any, float]] = None):
        """
        Initialize the LWW-Map.
        
        Args:
            node_id: Unique identifier for the node
            data: Optional initial data dictionary (key -> (value, timestamp))
        """
        super().__init__(node_id)
        self.data: Dict[str, Tuple[Any, float]] = {}
        if data:
            for key, (value, timestamp) in data.items():
                self.data[key] = (value, timestamp)
    
    def set(self, key: str, value: Any, timestamp: Optional[float] = None) -> None:
        """
        Set a key-value pair if the timestamp is newer.
        
        Args:
            key: Key to set
            value: Value to associate with the key
            timestamp: Timestamp for this write (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Only update if key doesn't exist or new timestamp is newer
        if key not in self.data or timestamp >= self.data[key][1]:
            self.data[key] = (value, timestamp)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get the value for a key.
        
        Args:
            key: Key to look up
            default: Default value if key doesn't exist
            
        Returns:
            Value for the key (not the timestamp), or default if not found
        """
        if key in self.data:
            return self.data[key][0]
        return default
    
    def keys(self) -> Set[str]:
        """
        Get all keys in the map.
        
        Returns:
            Set of all keys
        """
        return set(self.data.keys())
    
    def merge(self, other: 'LWWMap') -> 'LWWMap':
        """
        Merge this LWW-Map with another LWW-Map.
        
        Keeps the entry with the latest timestamp for each key.
        If timestamps are equal, keeps self's entry.
        
        Args:
            other: Another LWWMap instance
            
        Returns:
            A new LWWMap with merged entries
        """
        merged = LWWMap(self.node_id)
        
        # Get union of all keys
        all_keys = set(self.data.keys()) | set(other.data.keys())
        
        # For each key, keep the entry with the newer timestamp
        for key in all_keys:
            self_entry = self.data.get(key)
            other_entry = other.data.get(key)
            
            if self_entry is None:
                merged.data[key] = other_entry
            elif other_entry is None:
                merged.data[key] = self_entry
            else:
                # Both have the key - compare timestamps
                # If equal, prefer self's entry
                if other_entry[1] > self_entry[1]:
                    merged.data[key] = other_entry
                else:
                    merged.data[key] = self_entry
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the LWW-Map to a dictionary.
        
        Returns:
            Dictionary representation of the LWW-Map
        """
        serialized_data = {
            key: {'value': value, 'timestamp': timestamp}
            for key, (value, timestamp) in self.data.items()
        }
        return {
            'type': 'LWWMap',
            'node_id': self.node_id,
            'data': serialized_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LWWMap':
        """
        Create an LWWMap from a dictionary representation.
        
        Args:
            data: Dictionary containing serialized LWWMap state
            
        Returns:
            A new LWWMap instance
        """
        parsed_data = {}
        for key, entry in data.get('data', {}).items():
            parsed_data[key] = (entry['value'], entry['timestamp'])
        
        return cls(
            node_id=data['node_id'],
            data=parsed_data
        )
    
    def __repr__(self) -> str:
        keys_list = list(self.keys())
        preview = keys_list[:5]
        more = f", ... +{len(keys_list) - 5} more" if len(keys_list) > 5 else ""
        return f"LWWMap(size={len(keys_list)}, keys={preview}{more})"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __contains__(self, key: str) -> bool:
        return key in self.data
