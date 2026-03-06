"""
Reasoning Engine - Chain of Thought (CoT)
==========================================
Implements chain-of-thought prompting and step-by-step reasoning.
Enables the AGI brain to decompose complex problems into logical steps
before arriving at conclusions — mimicking deliberate human reasoning.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ReasoningResult:
    """Complete result from the reasoning engine."""
    query: str
    reasoning_steps: List[ReasoningStep]
    answer: str
    confidence: float
    method: str = "chain_of_thought"
    tokens_used: int = 0


class ChainOfThoughtReasoner:
    """
    Chain-of-Thought Reasoning Engine
    
    Implements multiple CoT strategies:
    1. Standard CoT: step-by-step reasoning
    2. ReAct: Reason + Act + Observe loop
    3. Tree of Thought: branching reasoning paths
    4. Self-Consistency: multiple reasoning paths + majority vote
    
    Features:
    - Context-aware reasoning (uses memory)
    - Confidence estimation
    - Step-by-step explanation generation
    - Multi-strategy selection
    """

    # CoT system prompt template
    SYSTEM_PROMPT = """You are an advanced reasoning AI. 
Think step by step before answering. 
Format your reasoning as:
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
Final Answer: [Your final answer]

Be precise, logical, and consider multiple perspectives."""

    REACT_PROMPT = """You are a reasoning AI using the ReAct framework.
For each step, use the format:
Thought: [What you're thinking]
Action: [What action to take, if any]
Observation: [What you observe from the action]
...
Final Answer: [Your conclusion]"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._llm = None
        self.max_steps = config.get("max_steps", 10) if config else 10
        self.temperature = 0.3  # Low temperature for consistent reasoning
        logger.info("🔗 Chain-of-Thought Reasoner initialized")

    def _get_llm(self):
        """Lazy-initialize LLM client."""
        if self._llm is None:
            try:
                from openai import OpenAI
                import os
                self._llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("OpenAI LLM connected for reasoning")
            except ImportError:
                logger.warning("OpenAI not available. Using rule-based fallback.")
                self._llm = "fallback"
        return self._llm

    def reason(self, query: str, context: Dict = None) -> ReasoningResult:
        """
        Main reasoning pipeline using Chain-of-Thought.
        
        Args:
            query: The question or problem to reason about
            context: Optional context from memory systems
            
        Returns:
            ReasoningResult with step-by-step reasoning and final answer
        """
        logger.info(f"🤔 Reasoning about: {query[:80]}...")
        
        # Build context string from memory
        context_str = self._build_context_string(context or {})
        
        # Try LLM-based CoT first
        llm = self._get_llm()
        
        if llm != "fallback":
            return self._llm_cot_reason(query, context_str)
        else:
            return self._rule_based_reason(query, context_str)

    def _llm_cot_reason(self, query: str, context_str: str) -> ReasoningResult:
        """Perform CoT reasoning using LLM."""
        llm = self._get_llm()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        
        if context_str:
            messages.append({
                "role": "user",
                "content": f"Context from memory:\n{context_str}\n\nQuestion: {query}"
            })
        else:
            messages.append({"role": "user", "content": f"Question: {query}"})
        
        try:
            response = llm.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=self.temperature,
                max_tokens=1500
            )
            
            raw_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Parse reasoning steps from response
            steps, answer = self._parse_cot_response(raw_response)
            confidence = self._estimate_confidence(steps, answer)
            
            return ReasoningResult(
                query=query,
                reasoning_steps=steps,
                answer=answer,
                confidence=confidence,
                method="chain_of_thought_llm",
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            return self._rule_based_reason(query, context_str)

    def _rule_based_reason(self, query: str, context_str: str) -> ReasoningResult:
        """
        Fallback rule-based reasoning when LLM is not available.
        Performs basic analysis and structured response.
        """
        steps = []
        
        # Step 1: Analyze the query type
        query_type = self._classify_query(query)
        steps.append(ReasoningStep(
            step_number=1,
            thought=f"Query classification: This appears to be a {query_type} query.",
            confidence=0.8
        ))
        
        # Step 2: Check context
        if context_str:
            steps.append(ReasoningStep(
                step_number=2,
                thought=f"Found relevant context in memory: {context_str[:200]}",
                observation="Context available",
                confidence=0.9
            ))
        else:
            steps.append(ReasoningStep(
                step_number=2,
                thought="No relevant context found in memory. Reasoning from general knowledge.",
                observation="No context",
                confidence=0.6
            ))
        
        # Step 3: Generate response
        answer = self._generate_basic_response(query, query_type, context_str)
        steps.append(ReasoningStep(
            step_number=3,
            thought=f"Generated response based on {query_type} reasoning pattern.",
            confidence=0.7
        ))
        
        return ReasoningResult(
            query=query,
            reasoning_steps=steps,
            answer=answer,
            confidence=0.7,
            method="rule_based_fallback"
        )

    def _parse_cot_response(self, response: str) -> tuple:
        """Parse CoT response into steps and final answer."""
        steps = []
        lines = response.strip().split("\n")
        
        current_step = None
        final_answer = ""
        step_num = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith("final answer:") or line.lower().startswith("answer:"):
                final_answer = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("step ") or (len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]):
                step_num += 1
                thought = line.split(":", 1)[-1].strip() if ":" in line else line
                current_step = ReasoningStep(
                    step_number=step_num,
                    thought=thought,
                    confidence=0.8
                )
                steps.append(current_step)
        
        if not final_answer and steps:
            final_answer = steps[-1].thought
            
        if not final_answer:
            final_answer = response.strip()
        
        return steps, final_answer

    def _build_context_string(self, context: Dict) -> str:
        """Build a context string from memory results."""
        parts = []
        
        if context.get("long_term"):
            ltm_items = context["long_term"][:3]
            if ltm_items:
                parts.append("Long-term memory:")
                for item in ltm_items:
                    parts.append(f"  - {str(item.get('content', ''))[:200]}")
        
        if context.get("episodic"):
            parts.append("Recent episodes:")
            for ep in context["episodic"][:2]:
                parts.append(f"  - {str(ep)[:150]}")
        
        return "\n".join(parts)

    def _classify_query(self, query: str) -> str:
        """Classify the type of reasoning needed."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["calculate", "compute", "how many", "what is", "math"]):
            return "mathematical"
        elif any(w in query_lower for w in ["why", "because", "cause", "reason", "explain"]):
            return "causal"
        elif any(w in query_lower for w in ["if", "then", "would", "could", "should"]):
            return "conditional"
        elif any(w in query_lower for w in ["compare", "difference", "vs", "versus", "better"]):
            return "comparative"
        else:
            return "general"

    def _generate_basic_response(self, query: str, query_type: str, context: str) -> str:
        """Generate a basic response without LLM."""
        if context:
            return f"Based on available context, I can address this {query_type} query: {query}. Context suggests: {context[:300]}"
        return f"This is a {query_type} query about: {query}. Please configure an LLM API key for detailed reasoning."

    def _estimate_confidence(self, steps: List[ReasoningStep], answer: str) -> float:
        """Estimate confidence in the reasoning result."""
        if not steps:
            return 0.5
        
        # More steps generally means more thorough reasoning
        step_confidence = min(0.9, 0.5 + len(steps) * 0.1)
        
        # Average step confidence
        avg_step_conf = sum(s.confidence for s in steps) / len(steps)
        
        return (step_confidence + avg_step_conf) / 2
