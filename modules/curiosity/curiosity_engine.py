"""
Curiosity System - Curiosity Engine
=====================================
Implements intrinsic motivation and novelty-driven exploration.
The AI becomes curious about new information it hasn't encountered before.

Inspired by:
- Intrinsic Curiosity Module (ICM) by Pathak et al.
- Count-Based Exploration methods
- Random Network Distillation (RND)
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class CuriosityEngine:
    """
    Curiosity & Novelty Detection Engine
    
    Core Concepts:
    - Novelty: How different is new input from past experiences?
    - Intrinsic Reward: AI is rewarded for exploring novel inputs
    - Exploration Rate: Controls the balance between exploration vs exploitation
    
    How it works:
    1. Embed the new input into vector space
    2. Compare with embeddings of past experiences
    3. If distance > threshold → novel → generate intrinsic reward
    4. Encourage the AGI to explore novel areas
    
    Analogous to: Human curiosity — attracted to new, unexpected things
    """

    def __init__(self, config: Dict = None, long_term_memory=None):
        self.config = config or {}
        self.long_term_memory = long_term_memory
        
        self.novelty_threshold = self.config.get("novelty_threshold", 0.3)
        self.exploration_rate = self.config.get("exploration_rate", 0.1)
        self.intrinsic_reward_scale = self.config.get("intrinsic_reward_scale", 0.5)
        
        # Track exploration history
        self._exploration_history: List[Dict] = []
        self._novelty_scores: List[float] = []
        self._total_intrinsic_reward = 0.0
        
        # Embedding cache for efficiency
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedder = None
        
        logger.info(f"🔍 Curiosity Engine initialized (threshold={self.novelty_threshold})")

    def _get_embedder(self):
        """Lazy-initialize sentence transformer."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get("memory_embedding_model", "all-MiniLM-L6-v2")
                self._embedder = SentenceTransformer(model_name)
            except ImportError:
                logger.warning("sentence-transformers not available for curiosity engine")
        return self._embedder

    def _embed(self, text: str) -> Optional[List[float]]:
        """Get or compute embedding for text."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedder = self._get_embedder()
        if embedder:
            embedding = embedder.encode(text).tolist()
            self._embedding_cache[text] = embedding
            return embedding
        return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def compute_novelty(self, text: str, top_k: int = 5) -> float:
        """
        Compute the novelty score of new input.
        
        Novelty = 1 - max(similarity with past experiences)
        Higher novelty = more different from anything seen before
        
        Args:
            text: Input text to evaluate
            top_k: Number of past memories to compare against
            
        Returns:
            float: Novelty score between 0.0 (very familiar) and 1.0 (completely novel)
        """
        embedding = self._embed(text)
        
        if embedding is None:
            # Fallback: random novelty with exploration rate
            return random.uniform(0.2, 0.8)
        
        # Retrieve similar memories from long-term storage
        if self.long_term_memory:
            try:
                similar = self.long_term_memory.retrieve(text, top_k=top_k)
                
                if not similar:
                    # Nothing similar found → completely novel
                    novelty = 1.0
                else:
                    # Max similarity across retrieved memories
                    max_similarity = max(
                        item.get("similarity", 0.0) for item in similar
                    )
                    novelty = 1.0 - max_similarity
            except Exception as e:
                logger.warning(f"Curiosity: Could not retrieve memories: {e}")
                novelty = 0.5
        else:
            # No memory available → treat everything as novel initially
            novelty = 0.8
        
        # Track novelty history
        self._novelty_scores.append(novelty)
        logger.debug(f"Novelty score: {novelty:.3f} (threshold: {self.novelty_threshold})")
        
        return novelty

    def get_intrinsic_reward(self, novelty_score: float) -> float:
        """
        Compute intrinsic reward from novelty score.
        
        High novelty → high intrinsic reward → encourages exploration
        
        Args:
            novelty_score: Novelty score from compute_novelty()
            
        Returns:
            float: Intrinsic reward signal
        """
        if novelty_score < self.novelty_threshold:
            return 0.0
        
        # Scale reward by novelty and the configured scale factor
        reward = novelty_score * self.intrinsic_reward_scale
        
        self._total_intrinsic_reward += reward
        logger.debug(f"Intrinsic reward: {reward:.3f}")
        
        return reward

    def should_explore(self) -> bool:
        """
        Epsilon-greedy exploration policy.
        Returns True if the agent should explore (random action).
        """
        return random.random() < self.exploration_rate

    def detect_interesting_patterns(self, text: str) -> Dict[str, Any]:
        """
        Identify what makes this input potentially interesting.
        
        Looks for:
        - Unexpected information (high novelty)
        - Contradictions with known information
        - Questions that need investigation
        - Unusual combinations of known concepts
        """
        novelty = self.compute_novelty(text)
        
        patterns = {
            "novelty_score": novelty,
            "is_novel": novelty > self.novelty_threshold,
            "interest_level": "high" if novelty > 0.7 else "medium" if novelty > 0.3 else "low",
            "recommended_action": "explore" if novelty > self.novelty_threshold else "exploit",
            "questions_to_investigate": self._generate_curiosity_questions(text, novelty),
        }
        
        # Log exploration event
        self._exploration_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "text_preview": text[:100],
            "novelty": novelty,
            "action": patterns["recommended_action"]
        })
        
        return patterns

    def _generate_curiosity_questions(self, text: str, novelty: float) -> List[str]:
        """Generate investigation questions based on novelty."""
        if novelty < self.novelty_threshold:
            return []
        
        questions = []
        
        # Basic curiosity questions based on content
        if "?" not in text:
            questions.append(f"What are the implications of: {text[:50]}?")
        
        if novelty > 0.7:
            questions.append("Why is this different from what I've seen before?")
            questions.append("What related topics should I learn about this?")
        
        if novelty > 0.5:
            questions.append("How does this connect to existing knowledge?")
        
        return questions[:3]  # Return max 3 questions

    def adapt_exploration_rate(self, performance_signal: float) -> None:
        """
        Adapt exploration rate based on performance.
        If performance is high → reduce exploration (exploit more)
        If performance is low → increase exploration
        
        Args:
            performance_signal: 0.0-1.0, where 1.0 = perfect performance
        """
        if performance_signal > 0.8:
            self.exploration_rate = max(0.01, self.exploration_rate * 0.95)
        elif performance_signal < 0.4:
            self.exploration_rate = min(0.5, self.exploration_rate * 1.05)
        
        logger.debug(f"Exploration rate adapted to: {self.exploration_rate:.3f}")

    def get_stats(self) -> Dict:
        """Get curiosity engine statistics."""
        return {
            "novelty_threshold": self.novelty_threshold,
            "exploration_rate": self.exploration_rate,
            "total_intrinsic_reward": self._total_intrinsic_reward,
            "avg_novelty": sum(self._novelty_scores) / len(self._novelty_scores) if self._novelty_scores else 0,
            "explorations_count": len(self._exploration_history),
            "high_novelty_events": sum(1 for s in self._novelty_scores if s > self.novelty_threshold),
        }
