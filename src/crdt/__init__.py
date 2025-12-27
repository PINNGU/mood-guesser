"""CRDT module for conflict-free replicated data types."""

from .base import CRDT
from .g_counter import GCounter

__all__ = ['CRDT', 'GCounter']
