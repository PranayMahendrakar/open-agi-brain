"""
Memory System - Episodic Memory
=================================
Stores and retrieves specific experiences and past events.
Each episode captures what happened, when it happened, and the outcome.

Analogous to: Human episodic memory — autobiographical memory of specific
events, experiences, and episodes in time.

"I remember asking that question about AI yesterday and getting this response."
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)


class Episode:
    """A single episodic memory entry."""
    
    def __init__(
        self,
        experience: Dict[str, Any],
        episode_id: str = None,
        timestamp: datetime = None
    ):
        self.episode_id = episode_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.utcnow()
        self.experience = experience
        self.recall_count = 0
        self.importance = self._compute_importance(experience)
    
    def _compute_importance(self, experience: Dict) -> float:
        """Estimate episode importance based on content."""
        importance = 0.5  # Default
        
        # Novel experiences are more important
        if experience.get("novelty", 0) > 0.5:
            importance += 0.2
        
        # High-confidence experiences are more important
        confidence = experience.get("confidence", 0.5)
        importance += confidence * 0.2
        
        return min(1.0, importance)
    
    def to_dict(self) -> Dict:
        """Convert episode to serializable dict."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp.isoformat(),
            "experience": {
                k: str(v)[:500] if not isinstance(v, (int, float, bool)) else v
                for k, v in self.experience.items()
            },
            "recall_count": self.recall_count,
            "importance": self.importance,
        }


class EpisodicMemory:
    """
    Episodic Memory — Experience & Event Storage
    
    Stores specific past experiences with:
    - What happened (input, response, action)
    - When it happened (timestamp)
    - Outcome metadata (novelty, confidence)
    
    Features:
    - Temporal ordering (recent first)
    - Importance-based retention
    - Time-based decay (optional)
    - Persistent storage (ChromaDB or in-memory)
    """

    def __init__(
        self,
        db_path: str = "./data/chroma_db",
        collection_name: str = "episodic_memory",
        max_episodes: int = 1000
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.max_episodes = max_episodes
        
        # In-memory fallback
        self._episodes: List[Episode] = []
        self._use_vector_db = False
        self._collection = None
        
        self._init_storage()
        logger.info(f"📚 Episodic Memory initialized (max={max_episodes})")

    def _init_storage(self):
        """Initialize storage backend."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.db_path)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._use_vector_db = True
            logger.debug("Episodic memory: ChromaDB backend active")
        except Exception:
            self._use_vector_db = False
            logger.debug("Episodic memory: In-memory backend active")

    def store_episode(self, experience: Dict[str, Any]) -> str:
        """
        Store a new episode/experience.
        
        Args:
            experience: Dict containing the experience data
                Keys: input, response, action, reasoning, novelty, confidence, etc.
                
        Returns:
            str: Episode ID
        """
        episode = Episode(experience=experience)
        
        if self._use_vector_db and self._collection:
            try:
                # Create searchable text for the episode
                episode_text = f"{experience.get('input', '')} {experience.get('response', '')}"[:1000]
                
                self._collection.add(
                    ids=[episode.episode_id],
                    documents=[episode_text],
                    metadatas=[{
                        "timestamp": episode.timestamp.isoformat(),
                        "importance": episode.importance,
                        "episode_json": json.dumps(episode.to_dict())[:2000]
                    }]
                )
            except Exception as e:
                logger.warning(f"Episode vector store failed: {e}, using in-memory")
                self._use_vector_db = False
                self._episodes.append(episode)
        else:
            self._episodes.append(episode)
            
            # Prune if over capacity
            if len(self._episodes) > self.max_episodes:
                self._prune_episodes()
        
        logger.debug(f"Episode stored: {episode.episode_id[:8]}...")
        return episode.episode_id

    def retrieve_recent(self, n: int = 5) -> List[Dict]:
        """
        Retrieve the n most recent episodes.
        
        Args:
            n: Number of episodes to retrieve
            
        Returns:
            List of episode dicts, newest first
        """
        if self._use_vector_db and self._collection:
            try:
                count = self._collection.count()
                if count == 0:
                    return []
                
                # Get all episodes and sort by timestamp (workaround for ChromaDB)
                results = self._collection.get(
                    limit=min(n * 2, count),
                    include=["metadatas"]
                )
                
                episodes = []
                for meta in results.get("metadatas", []):
                    try:
                        ep_dict = json.loads(meta.get("episode_json", "{}"))
                        episodes.append(ep_dict)
                    except Exception:
                        pass
                
                # Sort by timestamp
                episodes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return episodes[:n]
                
            except Exception as e:
                logger.warning(f"Episode retrieval failed: {e}")
                return []
        else:
            recent = sorted(self._episodes, key=lambda e: e.timestamp, reverse=True)
            return [e.to_dict() for e in recent[:n]]

    def search_by_content(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search episodes by content similarity."""
        if self._use_vector_db and self._collection:
            try:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                embedding = embedder.encode(query).tolist()
                
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=min(top_k, self._collection.count())
                )
                
                episodes = []
                for meta in results.get("metadatas", [[]])[0]:
                    try:
                        ep_dict = json.loads(meta.get("episode_json", "{}"))
                        episodes.append(ep_dict)
                    except Exception:
                        pass
                
                return episodes
                
            except Exception:
                pass
        
        # Fallback: simple text search
        query_lower = query.lower()
        matching = []
        for episode in reversed(self._episodes):
            ep_str = json.dumps(episode.experience).lower()
            if query_lower in ep_str:
                matching.append(episode.to_dict())
                if len(matching) >= top_k:
                    break
        
        return matching

    def get_episodes_by_timerange(
        self,
        start: datetime,
        end: datetime = None
    ) -> List[Dict]:
        """Retrieve episodes within a time range."""
        end = end or datetime.utcnow()
        
        episodes = self.retrieve_recent(n=100)
        filtered = [
            ep for ep in episodes
            if start.isoformat() <= ep.get("timestamp", "") <= end.isoformat()
        ]
        
        return filtered

    def _prune_episodes(self) -> None:
        """Remove least important episodes when over capacity."""
        # Sort by importance (keep most important)
        self._episodes.sort(key=lambda e: e.importance, reverse=True)
        self._episodes = self._episodes[:self.max_episodes]
        logger.debug(f"Episodic memory pruned to {len(self._episodes)} episodes")

    def count(self) -> int:
        """Return total number of stored episodes."""
        if self._use_vector_db and self._collection:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._episodes)

    def clear(self) -> None:
        """Clear all episodes."""
        self._episodes.clear()
        if self._use_vector_db and self._collection:
            try:
                self._collection.delete(where={})
            except Exception:
                pass
        logger.warning("All episodic memories cleared!")

    def __repr__(self) -> str:
        return f"EpisodicMemory(episodes={self.count()}, max={self.max_episodes})"
