"""
Unit tests for CRDT implementations.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.crdt.g_counter import GCounter
from src.crdt.or_set import ORSet
from src.crdt.lww_map import LWWMap
from src.crdt.state_manager import CRDTStateManager


class TestGCounter:
    """Tests for G-Counter CRDT."""
    
    def test_increment_and_value(self):
        """Test basic increment and value operations."""
        counter = GCounter("node1")
        assert counter.value() == 0
        
        counter.increment(5)
        assert counter.value() == 5
        
        counter.increment(3)
        assert counter.value() == 8
    
    def test_two_counters_merge(self):
        """Create two counters, increment each, merge, and verify sum."""
        counter1 = GCounter("node1")
        counter2 = GCounter("node2")
        
        counter1.increment(10)
        counter2.increment(7)
        
        # Merge counter2 into counter1
        merged = counter1.merge(counter2)
        
        # Merged value should be sum of both
        assert merged.value() == 17
    
    def test_merge_with_overlapping_increments(self):
        """Test merging counters that both have increments."""
        counter1 = GCounter("node1")
        counter2 = GCounter("node2")
        
        # Node1 increments itself
        counter1.increment(5)
        
        # Node2 increments itself
        counter2.increment(3)
        
        # Simulate node1 also incremented on node2's view
        counter2.counts["node1"] = 2
        
        # Merge should take max for each node
        merged = counter1.merge(counter2)
        
        # node1: max(5, 2) = 5, node2: max(0, 3) = 3
        assert merged.value() == 8
        assert merged.counts["node1"] == 5
        assert merged.counts["node2"] == 3
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        counter = GCounter("node1")
        counter.increment(10)
        
        data = counter.to_dict()
        
        assert data['type'] == 'GCounter'
        assert data['node_id'] == 'node1'
        assert data['counts']['node1'] == 10
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            'type': 'GCounter',
            'node_id': 'node1',
            'counts': {'node1': 5, 'node2': 3}
        }
        
        counter = GCounter.from_dict(data)
        
        assert counter.node_id == 'node1'
        assert counter.value() == 8
        assert counter.counts['node1'] == 5
        assert counter.counts['node2'] == 3
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = GCounter("node1")
        original.increment(15)
        
        data = original.to_dict()
        restored = GCounter.from_dict(data)
        
        assert restored.node_id == original.node_id
        assert restored.value() == original.value()
        assert restored.counts == original.counts
    
    def test_increment_negative_raises_error(self):
        """Test that incrementing with negative value raises error."""
        counter = GCounter("node1")
        
        with pytest.raises(ValueError):
            counter.increment(-1)


class TestORSet:
    """Tests for OR-Set CRDT."""
    
    def test_add_and_contains(self):
        """Test basic add and contains operations."""
        or_set = ORSet("node1")
        
        assert not or_set.contains("apple")
        
        or_set.add("apple")
        assert or_set.contains("apple")
    
    def test_values(self):
        """Test getting all values from the set."""
        or_set = ORSet("node1")
        or_set.add("apple")
        or_set.add("banana")
        or_set.add("cherry")
        
        values = or_set.values()
        assert values == {"apple", "banana", "cherry"}
    
    def test_two_sets_merge(self):
        """Create two sets, add different elements, merge, and verify all elements."""
        set1 = ORSet("node1")
        set2 = ORSet("node2")
        
        set1.add("apple")
        set1.add("banana")
        
        set2.add("cherry")
        set2.add("date")
        
        # Merge set2 into set1
        merged = set1.merge(set2)
        
        # Merged set should contain all elements
        assert merged.contains("apple")
        assert merged.contains("banana")
        assert merged.contains("cherry")
        assert merged.contains("date")
        assert merged.values() == {"apple", "banana", "cherry", "date"}
    
    def test_add_same_element_twice(self):
        """Test adding the same element multiple times."""
        or_set = ORSet("node1")
        
        or_set.add("apple")
        or_set.add("apple")
        
        assert or_set.contains("apple")
        # Should have multiple tags for the same element
        assert len(or_set.elements["apple"]) == 2
    
    def test_remove(self):
        """Test removing an element."""
        or_set = ORSet("node1")
        
        or_set.add("apple")
        or_set.add("banana")
        
        assert or_set.contains("apple")
        
        or_set.remove("apple")
        
        assert not or_set.contains("apple")
        assert or_set.contains("banana")
    
    def test_remove_nonexistent(self):
        """Test removing a non-existent element does not raise error."""
        or_set = ORSet("node1")
        or_set.remove("nonexistent")  # Should not raise
    
    def test_add_after_remove(self):
        """Test that adding after removing works correctly."""
        or_set = ORSet("node1")
        
        or_set.add("apple")
        or_set.remove("apple")
        or_set.add("apple")
        
        assert or_set.contains("apple")
    
    def test_merge_with_concurrent_add_remove(self):
        """Test merging when one node added and other removed."""
        set1 = ORSet("node1")
        set2 = ORSet("node2")
        
        # Both start with apple
        set1.add("apple")
        set2.elements["apple"] = set1.elements["apple"].copy()
        
        # Node1 removes apple
        set1.remove("apple")
        
        # Node2 adds apple again (new tag)
        set2.add("apple")
        
        # After merge, apple should exist (add wins due to new tag)
        merged = set1.merge(set2)
        assert merged.contains("apple")


class TestLWWMap:
    """Tests for LWW-Map CRDT."""
    
    def test_set_and_get(self):
        """Test basic set and get operations."""
        lww_map = LWWMap("node1")
        
        lww_map.set("key1", "value1", timestamp=1.0)
        
        assert lww_map.get("key1") == "value1"
    
    def test_get_default(self):
        """Test getting a non-existent key returns default."""
        lww_map = LWWMap("node1")
        
        assert lww_map.get("nonexistent") is None
        assert lww_map.get("nonexistent", "default") == "default"
    
    def test_newer_value_wins(self):
        """Test that setting with newer timestamp overwrites."""
        lww_map = LWWMap("node1")
        
        lww_map.set("key1", "old_value", timestamp=1.0)
        lww_map.set("key1", "new_value", timestamp=2.0)
        
        assert lww_map.get("key1") == "new_value"
    
    def test_older_value_ignored(self):
        """Test that setting with older timestamp is ignored."""
        lww_map = LWWMap("node1")
        
        lww_map.set("key1", "new_value", timestamp=2.0)
        lww_map.set("key1", "old_value", timestamp=1.0)
        
        assert lww_map.get("key1") == "new_value"
    
    def test_merge_newer_value_wins(self):
        """Create two maps, set same key with different timestamps, merge, and verify newer wins."""
        map1 = LWWMap("node1")
        map2 = LWWMap("node2")
        
        map1.set("key1", "value_from_node1", timestamp=1.0)
        map2.set("key1", "value_from_node2", timestamp=2.0)
        
        # Merge map2 into map1
        merged = map1.merge(map2)
        
        # Node2's value should win (newer timestamp)
        assert merged.get("key1") == "value_from_node2"
    
    def test_merge_multiple_keys(self):
        """Test merging maps with multiple keys."""
        map1 = LWWMap("node1")
        map2 = LWWMap("node2")
        
        # Map1 has newer value for key1
        map1.set("key1", "node1_value", timestamp=2.0)
        map2.set("key1", "node2_value", timestamp=1.0)
        
        # Map2 has newer value for key2
        map1.set("key2", "node1_value", timestamp=1.0)
        map2.set("key2", "node2_value", timestamp=2.0)
        
        # Only map1 has key3
        map1.set("key3", "only_node1", timestamp=1.0)
        
        # Only map2 has key4
        map2.set("key4", "only_node2", timestamp=1.0)
        
        merged = map1.merge(map2)
        
        # Verify correct values
        assert merged.get("key1") == "node1_value"  # map1 had newer
        assert merged.get("key2") == "node2_value"  # map2 had newer
        assert merged.get("key3") == "only_node1"   # only in map1
        assert merged.get("key4") == "only_node2"   # only in map2
    
    def test_keys(self):
        """Test getting all keys from the map."""
        lww_map = LWWMap("node1")
        
        lww_map.set("key1", "value1", timestamp=1.0)
        lww_map.set("key2", "value2", timestamp=1.0)
        lww_map.set("key3", "value3", timestamp=1.0)
        
        assert lww_map.keys() == {"key1", "key2", "key3"}
    
    def test_merge_equal_timestamps(self):
        """Test merging when timestamps are equal (self's value should win)."""
        map1 = LWWMap("node1")
        map2 = LWWMap("node2")
        
        map1.set("key1", "node1_value", timestamp=1.0)
        map2.set("key1", "node2_value", timestamp=1.0)
        
        merged = map1.merge(map2)
        
        # Self's value wins on equal timestamp
        assert merged.get("key1") == "node1_value"


class TestCRDTStateManager:
    """Tests for CRDT State Manager."""
    
    def test_increment_mood_count(self):
        """Test incrementing mood counts."""
        manager = CRDTStateManager("node1")
        
        manager.increment_mood_count("happy", 5)
        manager.increment_mood_count("sad", 3)
        
        assert manager.get_mood_count("happy") == 5
        assert manager.get_mood_count("sad") == 3
    
    def test_get_mood_count_nonexistent(self):
        """Test getting count for non-existent mood returns 0."""
        manager = CRDTStateManager("node1")
        
        assert manager.get_mood_count("nonexistent") == 0
    
    def test_merge_mood_counts(self):
        """Create two managers, increment moods differently, merge, and verify counts."""
        manager1 = CRDTStateManager("node1")
        manager2 = CRDTStateManager("node2")
        
        # Node1 counts
        manager1.increment_mood_count("happy", 10)
        manager1.increment_mood_count("sad", 5)
        
        # Node2 counts
        manager2.increment_mood_count("happy", 7)
        manager2.increment_mood_count("angry", 3)
        
        # Merge manager2 into manager1
        manager1.merge(manager2)
        
        # Verify merged counts
        assert manager1.get_mood_count("happy") == 17  # 10 + 7
        assert manager1.get_mood_count("sad") == 5     # only node1
        assert manager1.get_mood_count("angry") == 3   # only node2
    
    def test_training_rounds(self):
        """Test training rounds counter."""
        manager = CRDTStateManager("node1")
        
        assert manager.get_training_rounds() == 0
        
        manager.increment_training_rounds()
        manager.increment_training_rounds()
        
        assert manager.get_training_rounds() == 2
    
    def test_total_samples(self):
        """Test total samples counter."""
        manager = CRDTStateManager("node1")
        
        assert manager.get_total_samples() == 0
        
        manager.increment_total_samples(100)
        manager.increment_total_samples(50)
        
        assert manager.get_total_samples() == 150
    
    def test_merge_training_rounds_and_samples(self):
        """Test merging training rounds and sample counts."""
        manager1 = CRDTStateManager("node1")
        manager2 = CRDTStateManager("node2")
        
        manager1.increment_training_rounds()
        manager1.increment_total_samples(100)
        
        manager2.increment_training_rounds()
        manager2.increment_training_rounds()
        manager2.increment_total_samples(200)
        
        manager1.merge(manager2)
        
        assert manager1.get_training_rounds() == 3   # 1 + 2
        assert manager1.get_total_samples() == 300   # 100 + 200
    
    def test_model_version(self):
        """Test model version metadata."""
        manager = CRDTStateManager("node1")
        
        assert manager.get_model_version() == 0
        
        manager.update_model_version(1)
        assert manager.get_model_version() == 1
    
    def test_get_summary(self):
        """Test get_summary method."""
        manager = CRDTStateManager("node1")
        
        manager.increment_mood_count("happy", 10)
        manager.increment_mood_count("sad", 5)
        manager.increment_training_rounds()
        manager.increment_total_samples(15)
        manager.update_model_version(1)
        
        summary = manager.get_summary()
        
        assert summary['node_id'] == 'node1'
        assert summary['training_rounds'] == 1
        assert summary['total_samples'] == 15
        assert summary['model_version'] == 1
        assert 'happy' in summary['mood_counts']
        assert summary['mood_counts']['happy'] == 10
        assert 'sad' in summary['mood_counts']
        assert summary['mood_counts']['sad'] == 5
    
    def test_merge_preserves_mood_tags(self):
        """Test that merging preserves mood tags from both managers."""
        manager1 = CRDTStateManager("node1")
        manager2 = CRDTStateManager("node2")
        
        manager1.increment_mood_count("happy", 1)
        manager2.increment_mood_count("sad", 1)
        
        manager1.merge(manager2)
        
        # Both mood tags should be present
        assert manager1.mood_tags.contains("happy")
        assert manager1.mood_tags.contains("sad")
    
    def test_merge_metadata(self):
        """Test that metadata merging uses LWW semantics."""
        manager1 = CRDTStateManager("node1")
        manager2 = CRDTStateManager("node2")
        
        # Set metadata with different timestamps
        manager1.metadata.set("model_version", 1, timestamp=1.0)
        manager2.metadata.set("model_version", 2, timestamp=2.0)
        
        manager1.merge(manager2)
        
        # Newer value should win
        assert manager1.get_model_version() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
