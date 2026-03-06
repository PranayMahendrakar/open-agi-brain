"""
Memory System - Long-Term Memory (LTM)
=========================================
Persistent long-term memory using vector databases.
Stores knowledge and experiences as semantic embeddings for
similarity-based retrieval — analogous to human long-term memory.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """
    Long-Term Memory using Vector Database (ChromaDB)
    
    Features:
    - Persistent storage across sessions
    - Semantic similarity-based retrieval
    - Automatic embedding generation
    - Metadata filtering
    - Memory consolidation support
    
    Analogous to: Human long-term declarative memory
    Storage: Unlimited (disk-backed)
    Retrieval: Semantic similarity search
    """

    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "long_term_memory"):
        self.db_path = db_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedder = None
        logger.info(f"🟢 Long-Term Memory initialized (db_path={db_path})")

    def _get_client(self):
        """Lazy-initialize ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.db_path)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"ChromaDB connected. Collection: {self.collection_name}")
            except ImportError:
                logger.warning("ChromaDB not installed. Using in-memory fallback.")
                self._client = "fallback"
                self._fallback_store = {}
        return self._client

    def _get_embedder(self):
        """Lazy-initialize sentence transformer."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed.")
        return self._embedder

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        embedder = self._get_embedder()
        if embedder:
            return embedder.encode(text).tolist()
        return None

    def store(self, key: str, value: Any, metadata: Dict = None) -> str:
        """
        Store a memory in long-term storage.
        
        Args:
            key: Input/query text (used for embedding)
            value: The value/response to store
            metadata: Optional metadata dict
            
        Returns:
            str: Memory ID
        """
        client = self._get_client()
        memory_id = str(uuid.uuid4())
        
        meta = {
            "timestamp": datetime.utcnow().isoformat(),
            "key": str(key)[:500],
            **(metadata or {})
        }

        if client != "fallback":
            embedding = self._embed(str(key))
            if embedding:
                self._collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[str(value)[:2000]],
                    metadatas=[meta]
                )
        else:
            self._fallback_store[memory_id] = {
                "key": str(key),
                "value": str(value),
                "metadata": meta
            }
        
        logger.debug(f"LTM: Stored memory {memory_id[:8]}...")
        return memory_id

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """
        Retrieve memories similar to the query.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of memory dicts with 'content', 'similarity', 'metadata'
        """
        client = self._get_client()

        if client != "fallback":
            try:
                embedding = self._embed(query)
                if not embedding:
                    return []
                
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=min(top_k, self._collection.count()),
                    include=["documents", "metadatas", "distances"]
                )

                memories = []
                if results and results["documents"]:
                    for doc, meta, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    ):
                        similarity = 1.0 - dist  # ChromaDB uses distance
                        if similarity >= threshold:
                            memories.append({
                                "content": doc,
                                "similarity": similarity,
                                "metadata": meta,
                            })
                
                logger.debug(f"LTM: Retrieved {len(memories)} memories for query")
                return memories
                
            except Exception as e:
                logger.error(f"LTM retrieval error: {e}")
                return []
        else:
            # Fallback: return most recent memories
            items = list(self._fallback_store.values())[-top_k:]
            return [{"content": item["value"], "similarity": 1.0, "metadata": item["metadata"]} for item in items]

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        client = self._get_client()
        if client != "fallback":
            try:
                self._collection.delete(ids=[memory_id])
                return True
            except Exception as e:
                logger.error(f"LTM delete error: {e}")
                return False
        else:
            return self._fallback_store.pop(memory_id, None) is not None

    def count(self) -> int:
        """Return total number of stored memories."""
        client = self._get_client()
        if client != "fallback":
            try:
                return self._collection.count()
            except Exception:
                return 0
        return len(self._fallback_store)

    def clear(self) -> None:
        """Clear all memories (use with caution!)."""
        client = self._get_client()
        if client != "fallback":
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self._fallback_store.clear()
        logger.warning("LTM: All memories cleared!")

    def consolidate(self, threshold: float = 0.95) -> int:
        """
        Memory consolidation: remove near-duplicate memories.
        Returns number of memories removed.
        """
        # TODO: Implement consolidation logic
        logger.info("LTM: Memory consolidation not yet implemented")
        return 0

    def __repr__(self) -> str:
        return f"LongTermMemory(db_path={self.db_path}, count={self.count()})"
