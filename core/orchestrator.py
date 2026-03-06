"""
Open AGI Brain - Cognitive Orchestrator
========================================
Central coordinator that routes signals between all cognitive modules,
mimicking how the prefrontal cortex integrates information across brain regions.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from modules.perception.text_processor import TextProcessor
from modules.perception.vision_processor import VisionProcessor
from modules.perception.audio_processor import AudioProcessor
from modules.reasoning.chain_of_thought import ChainOfThoughtReasoner
from modules.reasoning.symbolic import SymbolicReasoner
from modules.reasoning.causal import CausalReasoner
from modules.memory.short_term import ShortTermMemory
from modules.memory.working_memory import WorkingMemory
from modules.memory.long_term import LongTermMemory
from modules.memory.episodic import EpisodicMemory
from modules.memory.semantic import SemanticMemory
from modules.curiosity.curiosity_engine import CuriosityEngine
from modules.decision.decision_engine import DecisionEngine
from modules.self_reflection.reflection_module import SelfReflectionModule
from utils.logger import get_logger

logger = get_logger(__name__)


class CognitiveOrchestrator:
    """
    The Central Cognitive Orchestrator — the 'brain stem' of the AGI system.
    
    Coordinates all cognitive modules:
    - Perception (input processing)
    - Reasoning (logic & problem-solving)
    - Memory (storage & retrieval)
    - Curiosity (exploration & novelty)
    - Decision (action selection)
    - Self-Reflection (meta-cognition)
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        logger.info("🧠 Initializing Open AGI Brain Cognitive Orchestrator...")
        self._init_modules()
        logger.info("✅ All cognitive modules initialized successfully!")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file not found at {config_path}, using defaults.")
        return {}

    def _init_modules(self):
        """Initialize all cognitive modules."""
        # Perception Engine
        self.text_processor = TextProcessor(self.config.get("perception", {}))
        self.vision_processor = VisionProcessor(self.config.get("perception", {}))
        self.audio_processor = AudioProcessor(self.config.get("perception", {}))

        # Memory System
        self.short_term_memory = ShortTermMemory(
            capacity=self.config.get("memory", {}).get("short_term_capacity", 10)
        )
        self.working_memory = WorkingMemory(
            capacity=self.config.get("memory", {}).get("working_memory_capacity", 5)
        )
        self.long_term_memory = LongTermMemory(
            db_path=self.config.get("memory", {}).get("vector_db_path", "./data/chroma_db")
        )
        self.episodic_memory = EpisodicMemory(
            db_path=self.config.get("memory", {}).get("vector_db_path", "./data/chroma_db")
        )
        self.semantic_memory = SemanticMemory()

        # Reasoning Engine
        self.cot_reasoner = ChainOfThoughtReasoner(self.config.get("reasoning", {}))
        self.symbolic_reasoner = SymbolicReasoner()
        self.causal_reasoner = CausalReasoner()

        # Curiosity System
        self.curiosity_engine = CuriosityEngine(
            self.config.get("curiosity", {}),
            self.long_term_memory
        )

        # Decision Engine
        self.decision_engine = DecisionEngine(self.config.get("decision", {}))

        # Self-Reflection Module
        self.reflection_module = SelfReflectionModule(self.config.get("self_reflection", {}))

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main cognitive processing pipeline.
        
        Pipeline:
        1. Perception: Process raw input
        2. Memory: Retrieve relevant context
        3. Curiosity: Assess novelty
        4. Reasoning: Generate response
        5. Reflection: Evaluate & improve
        6. Decision: Select action
        7. Memory: Store experience
        
        Args:
            input_data: Dict with keys 'type' (text/image/audio) and 'content'
            
        Returns:
            Dict with 'response', 'reasoning', 'confidence', 'action'
        """
        logger.info(f"🔄 Processing input of type: {input_data.get('type', 'unknown')}")

        # Step 1: Perception — Process raw input
        perceived = self._perceive(input_data)
        logger.debug(f"Perceived: {str(perceived)[:100]}...")

        # Step 2: Memory — Store in short-term & retrieve long-term context
        self.short_term_memory.add(perceived)
        stm_context = self.short_term_memory.get_all()
        ltm_context = self.long_term_memory.retrieve(str(perceived), top_k=5)
        episodic_context = self.episodic_memory.retrieve_recent(n=3)

        # Step 3: Curiosity — Detect novelty and generate intrinsic reward
        novelty_score = self.curiosity_engine.compute_novelty(str(perceived))
        if novelty_score > self.config.get("curiosity", {}).get("novelty_threshold", 0.3):
            logger.info(f"🔍 Novel input detected! Novelty score: {novelty_score:.3f}")
            intrinsic_reward = self.curiosity_engine.get_intrinsic_reward(novelty_score)
        else:
            intrinsic_reward = 0.0

        # Step 4: Reasoning — Generate response using chain-of-thought
        context = {
            "short_term": stm_context,
            "long_term": ltm_context,
            "episodic": episodic_context,
            "semantic": self.semantic_memory.query(str(perceived)),
        }
        reasoning_result = self.cot_reasoner.reason(
            query=str(perceived),
            context=context
        )

        # Step 5: Self-Reflection — Evaluate and improve response
        if self.config.get("self_reflection", {}).get("enabled", True):
            final_response = self.reflection_module.reflect_and_improve(
                query=str(perceived),
                initial_answer=reasoning_result["answer"],
                context=context
            )
        else:
            final_response = reasoning_result["answer"]

        # Step 6: Decision — Select best action
        action = self.decision_engine.decide(
            state=perceived,
            response=final_response,
            context=context
        )

        # Step 7: Memory — Store experience in long-term & episodic memory
        experience = {
            "input": str(perceived),
            "response": final_response,
            "reasoning": reasoning_result.get("reasoning_steps", []),
            "action": action,
            "novelty": novelty_score,
        }
        self.long_term_memory.store(str(perceived), final_response)
        self.episodic_memory.store_episode(experience)

        result = {
            "response": final_response,
            "reasoning_steps": reasoning_result.get("reasoning_steps", []),
            "action": action,
            "confidence": reasoning_result.get("confidence", 0.0),
            "novelty_score": novelty_score,
            "intrinsic_reward": intrinsic_reward,
        }
        
        logger.info(f"✅ Processing complete. Confidence: {result['confidence']:.3f}")
        return result

    def _perceive(self, input_data: Dict[str, Any]) -> Any:
        """Route input to appropriate perception module."""
        input_type = input_data.get("type", "text")
        content = input_data.get("content", "")

        if input_type == "text":
            return self.text_processor.process(content)
        elif input_type == "image":
            return self.vision_processor.process(content)
        elif input_type == "audio":
            return self.audio_processor.process(content)
        else:
            logger.warning(f"Unknown input type: {input_type}, treating as text")
            return self.text_processor.process(str(content))

    def remember(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query all memory systems for relevant information."""
        return {
            "short_term": self.short_term_memory.get_all(),
            "long_term": self.long_term_memory.retrieve(query, top_k=top_k),
            "episodic": self.episodic_memory.retrieve_recent(n=top_k),
            "semantic": self.semantic_memory.query(query),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of all cognitive modules."""
        return {
            "short_term_memory_size": len(self.short_term_memory.get_all()),
            "working_memory_active": len(self.working_memory.get_active()),
            "semantic_knowledge_nodes": self.semantic_memory.node_count(),
            "modules_active": [
                "perception", "reasoning", "memory",
                "curiosity", "decision", "self_reflection"
            ]
        }
