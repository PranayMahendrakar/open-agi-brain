"""
Memory System - Short-Term Memory (STM)
=========================================
Implements a capacity-limited short-term memory buffer.
Analogous to human working short-term memory — holds recent perceptions
for immediate access but discards oldest items when full.
"""

from collections import deque
from typing import Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryItem:
    """A single memory item with content and metadata."""
    content: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 1.0
    access_count: int = 0

    def access(self):
        """Mark this item as accessed."""
        self.access_count += 1
        return self.content


class ShortTermMemory:
    """
    Short-Term Memory Buffer
    
    Features:
    - Fixed capacity with automatic eviction (oldest-first)
    - O(1) append and access operations
    - Importance-weighted retention (coming soon)
    - Temporal ordering preserved
    
    Analogous to: Human short-term/sensory memory
    Capacity: ~7 items (configurable)
    Duration: Session-scoped (cleared on reset)
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        logger.info(f"🔵 Short-Term Memory initialized (capacity={capacity})")

    def add(self, content: Any, importance: float = 1.0) -> None:
        """
        Add an item to short-term memory.
        
        If at capacity, automatically evicts the oldest item.
        
        Args:
            content: The content to store
            importance: Importance score (0.0-1.0) for future prioritization
        """
        item = MemoryItem(content=content, importance=importance)
        self._buffer.append(item)
        logger.debug(f"STM: Added item. Buffer size: {len(self._buffer)}/{self.capacity}")

    def get_all(self) -> List[Any]:
        """Return all items in memory (newest to oldest)."""
        return [item.content for item in reversed(self._buffer)]

    def get_latest(self, n: int = 1) -> List[Any]:
        """Get the n most recent items."""
        items = list(reversed(self._buffer))
        return [item.content for item in items[:n]]

    def get_oldest(self) -> Optional[Any]:
        """Get the oldest item in memory."""
        if self._buffer:
            return self._buffer[0].content
        return None

    def peek(self) -> Optional[Any]:
        """Get the most recent item without affecting order."""
        if self._buffer:
            return self._buffer[-1].content
        return None

    def clear(self) -> None:
        """Clear all items from short-term memory."""
        self._buffer.clear()
        logger.debug("STM: Memory cleared")

    def is_full(self) -> bool:
        """Check if memory is at capacity."""
        return len(self._buffer) >= self.capacity

    def size(self) -> int:
        """Return current number of items in memory."""
        return len(self._buffer)

    def contains(self, content: Any) -> bool:
        """Check if specific content exists in memory."""
        return any(str(item.content) == str(content) for item in self._buffer)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "capacity": self.capacity,
            "current_size": len(self._buffer),
            "utilization": len(self._buffer) / self.capacity,
            "oldest_item_age": (
                (datetime.utcnow() - self._buffer[0].timestamp).total_seconds()
                if self._buffer else 0
            ),
        }

    def __repr__(self) -> str:
        return f"ShortTermMemory(capacity={self.capacity}, size={len(self._buffer)})"

    def __len__(self) -> int:
        return len(self._buffer)
