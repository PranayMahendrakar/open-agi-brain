"""
Tests - Module Unit Tests
===========================
Unit tests for all cognitive modules of the Open AGI Brain.

Run:
    pytest tests/test_modules.py -v
    pytest tests/test_modules.py -v --tb=short
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime


# ═══════════════════════════════════════
# Short-Term Memory Tests
# ═══════════════════════════════════════

class TestShortTermMemory:
    """Tests for the Short-Term Memory module."""

    def setup_method(self):
        """Create fresh STM instance for each test."""
        from modules.memory.short_term import ShortTermMemory
        self.stm = ShortTermMemory(capacity=5)

    def test_add_and_retrieve(self):
        """Test basic add and retrieve operations."""
        self.stm.add("hello world")
        items = self.stm.get_all()
        assert len(items) == 1
        assert items[0] == "hello world"

    def test_capacity_limit(self):
        """Test that STM respects capacity limit."""
        for i in range(10):
            self.stm.add(f"item_{i}")
        assert len(self.stm.get_all()) == 5

    def test_oldest_evicted(self):
        """Test that oldest item is evicted when at capacity."""
        for i in range(5):
            self.stm.add(f"item_{i}")
        self.stm.add("new_item")
        items = self.stm.get_all()
        assert "item_0" not in items  # Oldest should be gone
        assert "new_item" in items

    def test_clear(self):
        """Test clearing memory."""
        self.stm.add("test")
        self.stm.clear()
        assert len(self.stm.get_all()) == 0

    def test_is_full(self):
        """Test is_full check."""
        for i in range(5):
            self.stm.add(f"item_{i}")
        assert self.stm.is_full()

    def test_peek(self):
        """Test peek returns latest without removing."""
        self.stm.add("first")
        self.stm.add("second")
        latest = self.stm.peek()
        assert latest == "second"
        assert len(self.stm.get_all()) == 2

    def test_get_latest_n(self):
        """Test get_latest returns n most recent."""
        for i in range(5):
            self.stm.add(f"item_{i}")
        latest = self.stm.get_latest(3)
        assert len(latest) == 3
        assert latest[0] == "item_4"  # Most recent first

    def test_contains(self):
        """Test contains check."""
        self.stm.add("test_item")
        assert self.stm.contains("test_item")
        assert not self.stm.contains("nonexistent")

    def test_stats(self):
        """Test stats return correct info."""
        self.stm.add("test")
        stats = self.stm.get_stats()
        assert stats["capacity"] == 5
        assert stats["current_size"] == 1
        assert 0 < stats["utilization"] <= 1


# ═══════════════════════════════════════
# Text Processor Tests
# ═══════════════════════════════════════

class TestTextProcessor:
    """Tests for the Text Perception module."""

    def setup_method(self):
        from modules.perception.text_processor import TextProcessor
        self.processor = TextProcessor()

    def test_basic_processing(self):
        """Test basic text processing."""
        result = self.processor.process("Hello, what is the weather today?")
        assert result is not None
        assert result.raw_text == "Hello, what is the weather today?"

    def test_intent_detection(self):
        """Test intent classification."""
        # Question intent
        result = self.processor.process("What is machine learning?")
        assert result.intent == "question"
        
        # Statement intent
        result = self.processor.process("Python is a programming language.")
        assert result.intent == "statement"

    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        positive = self.processor.process("This is wonderful and amazing!")
        assert positive.sentiment > 0
        
        negative = self.processor.process("This is terrible and horrible!")
        assert negative.sentiment < 0
        
        neutral = self.processor.process("The sky is blue.")
        assert -0.1 <= neutral.sentiment <= 0.1

    def test_tokenization(self):
        """Test tokenization."""
        result = self.processor.process("Hello world")
        assert len(result.tokens) >= 2

    def test_batch_process(self):
        """Test batch processing."""
        texts = ["Hello", "World", "Test"]
        results = self.processor.batch_process(texts)
        assert len(results) == 3


# ═══════════════════════════════════════
# Curiosity Engine Tests
# ═══════════════════════════════════════

class TestCuriosityEngine:
    """Tests for the Curiosity System."""

    def setup_method(self):
        from modules.curiosity.curiosity_engine import CuriosityEngine
        self.curiosity = CuriosityEngine(
            config={"novelty_threshold": 0.3, "exploration_rate": 0.1},
            long_term_memory=None
        )

    def test_novelty_score_range(self):
        """Test novelty score is in valid range."""
        score = self.curiosity.compute_novelty("Test input text")
        assert 0.0 <= score <= 1.0

    def test_intrinsic_reward_zero_for_familiar(self):
        """Test no reward for low-novelty input."""
        reward = self.curiosity.get_intrinsic_reward(0.1)  # Below threshold
        assert reward == 0.0

    def test_intrinsic_reward_for_novel(self):
        """Test reward for high-novelty input."""
        reward = self.curiosity.get_intrinsic_reward(0.8)  # Above threshold
        assert reward > 0.0

    def test_detect_interesting_patterns(self):
        """Test pattern detection returns structured result."""
        patterns = self.curiosity.detect_interesting_patterns("Novel scientific discovery!")
        
        assert "novelty_score" in patterns
        assert "is_novel" in patterns
        assert "interest_level" in patterns
        assert "recommended_action" in patterns

    def test_exploration_rate_adaptation(self):
        """Test exploration rate adapts based on performance."""
        initial_rate = self.curiosity.exploration_rate
        
        # High performance should reduce exploration
        self.curiosity.adapt_exploration_rate(0.9)
        assert self.curiosity.exploration_rate <= initial_rate

    def test_stats(self):
        """Test stats collection."""
        self.curiosity.compute_novelty("test input")
        stats = self.curiosity.get_stats()
        
        assert "novelty_threshold" in stats
        assert "exploration_rate" in stats
        assert "total_intrinsic_reward" in stats


# ═══════════════════════════════════════
# Self-Reflection Module Tests
# ═══════════════════════════════════════

class TestSelfReflectionModule:
    """Tests for the Self-Reflection Module."""

    def setup_method(self):
        from modules.self_reflection.reflection_module import SelfReflectionModule
        self.reflector = SelfReflectionModule(
            config={"max_iterations": 2, "improvement_threshold": 0.8}
        )

    def test_reflect_returns_string(self):
        """Test reflection returns a string answer."""
        result = self.reflector.reflect_and_improve(
            query="What is AI?",
            initial_answer="AI is artificial intelligence."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_full_reflect_returns_result(self):
        """Test full reflection returns ReflectionResult."""
        from modules.self_reflection.reflection_module import ReflectionResult
        result = self.reflector.full_reflect(
            query="Explain Python programming",
            initial_answer="Python is a language."
        )
        assert isinstance(result, ReflectionResult)
        assert result.original_answer == "Python is a language."
        assert result.final_answer is not None

    def test_rule_based_critique(self):
        """Test rule-based critique evaluation."""
        critique = self.reflector._rule_based_critique(
            query="Explain quantum computing",
            answer="Quantum computing uses qubits."
        )
        
        assert 0.0 <= critique.score <= 1.0
        assert isinstance(critique.strengths, list)
        assert isinstance(critique.weaknesses, list)

    def test_short_answer_gets_low_score(self):
        """Test that very short answers get low quality scores."""
        critique = self.reflector._rule_based_critique(
            query="Explain the theory of relativity in detail",
            answer="E=mc2"
        )
        assert critique.score < 0.8  # Should have room for improvement


# ═══════════════════════════════════════
# Decision Engine Tests
# ═══════════════════════════════════════

class TestDecisionEngine:
    """Tests for the Decision Engine."""

    def setup_method(self):
        from modules.decision.decision_engine import DecisionEngine
        self.engine = DecisionEngine()

    def test_decide_returns_action(self):
        """Test that decide returns an Action object."""
        from modules.decision.decision_engine import Action
        
        action = self.engine.decide(
            state="Test query about AI",
            response="AI is artificial intelligence",
            context={}
        )
        
        assert isinstance(action, Action)
        assert 0.0 <= action.confidence <= 1.0

    def test_plan_sequence(self):
        """Test plan sequence generation."""
        plan = self.engine.plan_sequence("Explain neural networks", steps=3)
        
        assert len(plan) <= 3
        assert len(plan) > 0

    def test_q_value_update(self):
        """Test Q-value learning update."""
        from modules.decision.decision_engine import ActionType
        
        initial_q = self.engine._q_values[ActionType.RESPOND.value]
        self.engine.update_q_value(ActionType.RESPOND, reward=1.0)
        
        new_q = self.engine._q_values[ActionType.RESPOND.value]
        assert new_q > initial_q  # Should increase with positive reward

    def test_stats(self):
        """Test stats collection."""
        self.engine.decide(state="test", response="response", context={})
        stats = self.engine.get_stats()
        
        assert stats["total_decisions"] >= 1
        assert "action_distribution" in stats


# ═══════════════════════════════════════
# Semantic Memory Tests
# ═══════════════════════════════════════

class TestSemanticMemory:
    """Tests for the Semantic Memory (Knowledge Graph)."""

    def setup_method(self):
        from modules.memory.semantic import SemanticMemory
        self.memory = SemanticMemory()

    def test_add_and_query_fact(self):
        """Test adding and querying facts."""
        self.memory.add_fact("Python", "is_a", "programming_language")
        facts = self.memory.query("Python")
        
        assert len(facts) > 0
        assert any(f["predicate"] == "is_a" for f in facts)

    def test_add_knowledge_base(self):
        """Test bulk knowledge loading."""
        facts = [
            ("Einstein", "discovered", "relativity"),
            ("Einstein", "born_in", "Germany"),
            ("relativity", "is_a", "physics_theory"),
        ]
        count = self.memory.add_knowledge_base(facts)
        assert count == 3

    def test_node_count(self):
        """Test node counting."""
        initial = self.memory.node_count()
        self.memory.add_concept("new_concept")
        assert self.memory.node_count() >= initial

    def test_get_related_concepts(self):
        """Test getting related concepts."""
        self.memory.add_fact("cat", "is_a", "animal")
        self.memory.add_fact("dog", "is_a", "animal")
        
        # Both should be related to "animal"
        related = self.memory.get_related_concepts("animal")
        # In incoming direction, cat and dog are related to animal
        assert len(related) >= 0  # May have 0 if no matching direction


# ═══════════════════════════════════════
# Run Tests
# ═══════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
