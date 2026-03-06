"""
Memory System - Working Memory
================================
Active workspace for current task processing.
Holds the information currently being manipulated or processed.

Analogous to: Human working memory — the "RAM" of cognition.
Limited capacity, fast access, holds current task context.

George Miller's "Magical Number 7 ± 2" — human working memory 
can hold about 7 items simultaneously.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WorkingMemorySlot:
    """A slot in working memory holding an active item."""
    slot_id: str
    content: Any
    label: str = ""
    priority: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    def access(self) -> Any:
        """Access the slot content, updating access metadata."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
        return self.content


class WorkingMemory:
    """
    Working Memory — Active Task Workspace
    
    Manages a limited set of "slots" for active information:
    - Current task context
    - Intermediate reasoning results
    - Active goals
    - Temporary variables
    
    Features:
    - Named slots for organized access
    - Priority-based management
    - LRU eviction when at capacity
    - Workspace operations (set, get, delete, clear)
    
    Capacity: 5-9 items (configurable)
    """

    def __init__(self, capacity: int = 7):
        """
        Initialize Working Memory.
        
        Args:
            capacity: Maximum number of slots (default: 7, Miller's magic number)
        """
        self.capacity = capacity
        self._slots: Dict[str, WorkingMemorySlot] = {}
        logger.info(f"🟡 Working Memory initialized (capacity={capacity})")

    def set(self, key: str, value: Any, priority: float = 1.0, label: str = "") -> bool:
        """
        Set a value in working memory.
        
        If at capacity, evicts the least recently used item.
        
        Args:
            key: Unique identifier for this slot
            value: Content to store
            priority: Priority level (higher = more important)
            label: Human-readable label for debugging
            
        Returns:
            bool: True if successfully stored
        """
        # If key already exists, update it
        if key in self._slots:
            self._slots[key].content = value
            self._slots[key].last_accessed = datetime.utcnow()
            self._slots[key].priority = priority
            return True
        
        # If at capacity, evict LRU item
        if len(self._slots) >= self.capacity:
            self._evict_lru()
        
        self._slots[key] = WorkingMemorySlot(
            slot_id=key,
            content=value,
            label=label or key,
            priority=priority
        )
        
        logger.debug(f"WM: Set '{key}' (slots: {len(self._slots)}/{self.capacity})")
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from working memory.
        
        Args:
            key: Slot identifier
            default: Value to return if key not found
            
        Returns:
            Slot content or default
        """
        if key in self._slots:
            return self._slots[key].access()
        return default

    def delete(self, key: str) -> bool:
        """
        Remove a slot from working memory.
        
        Returns:
            bool: True if slot existed and was removed
        """
        if key in self._slots:
            del self._slots[key]
            logger.debug(f"WM: Deleted '{key}'")
            return True
        return False

    def get_active(self) -> Dict[str, Any]:
        """Get all active working memory contents."""
        return {key: slot.content for key, slot in self._slots.items()}

    def set_goal(self, goal: str) -> None:
        """Set the current active goal."""
        self.set("_active_goal", goal, priority=2.0, label="Active Goal")

    def get_goal(self) -> Optional[str]:
        """Get the current active goal."""
        return self.get("_active_goal")

    def set_context(self, context: Dict) -> None:
        """Set the current task context."""
        self.set("_task_context", context, priority=1.5, label="Task Context")

    def get_context(self) -> Dict:
        """Get the current task context."""
        return self.get("_task_context", {})

    def push_reasoning_step(self, step: str) -> None:
        """Push a reasoning step to the reasoning buffer."""
        steps = self.get("_reasoning_steps", [])
        steps.append({"step": step, "timestamp": datetime.utcnow().isoformat()})
        self.set("_reasoning_steps", steps, priority=1.2, label="Reasoning Steps")

    def get_reasoning_steps(self) -> List[Dict]:
        """Get all stored reasoning steps."""
        return self.get("_reasoning_steps", [])

    def clear_reasoning(self) -> None:
        """Clear reasoning buffer."""
        self.delete("_reasoning_steps")

    def _evict_lru(self) -> Optional[str]:
        """
        Evict the least recently used slot.
        
        Considers priority and last access time.
        Returns the evicted key.
        """
        if not self._slots:
            return None
        
        # Find slot to evict (lowest priority, oldest access)
        # Score = priority * recency_factor
        now = datetime.utcnow()
        
        def eviction_score(slot: WorkingMemorySlot) -> float:
            age_seconds = (now - slot.last_accessed).total_seconds()
            recency = 1.0 / (1.0 + age_seconds)  # Higher = more recent
            return slot.priority * recency
        
        lru_key = min(self._slots.keys(), key=lambda k: eviction_score(self._slots[k]))
        
        logger.debug(f"WM: Evicting LRU slot '{lru_key}'")
        del self._slots[lru_key]
        return lru_key

    def is_full(self) -> bool:
        """Check if working memory is at capacity."""
        return len(self._slots) >= self.capacity

    def size(self) -> int:
        """Return current number of active slots."""
        return len(self._slots)

    def clear(self) -> None:
        """Clear all working memory slots."""
        self._slots.clear()
        logger.debug("WM: All slots cleared")

    def get_stats(self) -> Dict:
        """Get working memory statistics."""
        return {
            "capacity": self.capacity,
            "active_slots": len(self._slots),
            "utilization": len(self._slots) / self.capacity,
            "slot_keys": list(self._slots.keys()),
            "slot_priorities": {k: v.priority for k, v in self._slots.items()},
        }

    def __repr__(self) -> str:
        return f"WorkingMemory(capacity={self.capacity}, active={len(self._slots)})"

    def __contains__(self, key: str) -> bool:
        return key in self._slots

    def __len__(self) -> int:
        return len(self._slots)
