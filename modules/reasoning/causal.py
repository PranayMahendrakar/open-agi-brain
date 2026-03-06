"""
Reasoning Engine - Causal Inference
======================================
Implements causal reasoning — understanding cause and effect relationships.

Causal reasoning enables the AGI to:
- Identify causes of observed events
- Predict effects of actions
- Reason about counterfactuals ("What if X had not happened?")
- Build and query causal graphs

Inspired by:
- Pearl's Causal Hierarchy (Association, Intervention, Counterfactuals)
- Structural Causal Models (SCMs)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CausalRelation:
    """A causal relationship between two events/variables."""
    cause: str
    effect: str
    strength: float = 1.0      # How strongly cause leads to effect (0-1)
    confidence: float = 1.0    # Confidence in this relationship (0-1)
    mechanism: str = ""        # How/why the causation works
    is_bidirectional: bool = False

    def __repr__(self) -> str:
        direction = "↔" if self.is_bidirectional else "→"
        return f"{self.cause} {direction} {self.effect} (strength={self.strength:.2f})"


@dataclass
class CausalQuery:
    """A causal query and its result."""
    query_type: str           # "what_causes", "what_effects", "counterfactual"
    query: str
    answer: str
    causal_chain: List[str]   # The chain of causation
    confidence: float
    alternative_causes: List[str] = field(default_factory=list)


class CausalReasoner:
    """
    Causal Inference and Reasoning Engine
    
    Implements Pearl's Causal Hierarchy:
    1. Association (observational): P(Y | X) — correlation
    2. Intervention (do-calculus): P(Y | do(X)) — manipulation
    3. Counterfactual: P(Y_x | X=x') — what if?
    
    Features:
    - Build causal graphs from observations
    - Query: "What causes X?"
    - Query: "What are the effects of X?"
    - Counterfactual reasoning: "What if X had not happened?"
    - Causal chain tracing
    """

    def __init__(self):
        # Causal graph: cause → list of effects
        self._causal_graph: Dict[str, List[CausalRelation]] = {}
        # Reverse graph: effect → list of causes  
        self._reverse_graph: Dict[str, List[CausalRelation]] = {}
        
        # Pre-populate with some common causal knowledge
        self._init_default_knowledge()
        logger.info("🔀 Causal Reasoner initialized")

    def _init_default_knowledge(self):
        """Initialize with common causal relationships."""
        default_relations = [
            ("learning", "knowledge_gain", 0.95, "Learning leads to knowledge"),
            ("exercise", "improved_health", 0.85, "Exercise improves health"),
            ("data", "machine_learning", 0.9, "Data enables ML training"),
            ("machine_learning", "predictions", 0.9, "ML produces predictions"),
            ("code_errors", "bugs", 0.95, "Errors cause bugs"),
            ("testing", "quality_improvement", 0.8, "Testing improves quality"),
            ("training", "model_improvement", 0.9, "Training improves models"),
            ("feedback", "performance_improvement", 0.75, "Feedback drives improvement"),
        ]
        
        for cause, effect, strength, mechanism in default_relations:
            self.add_causal_relation(cause, effect, strength, mechanism=mechanism)

    def add_causal_relation(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        confidence: float = 1.0,
        mechanism: str = "",
        bidirectional: bool = False
    ) -> CausalRelation:
        """
        Add a causal relationship to the knowledge base.
        
        Args:
            cause: The causing event/variable
            effect: The resulting event/variable
            strength: Causal strength (0-1)
            confidence: Confidence in the relationship
            mechanism: Explanation of the causal mechanism
            bidirectional: Whether causation goes both ways
            
        Returns:
            CausalRelation object
        """
        relation = CausalRelation(
            cause=cause.lower().strip(),
            effect=effect.lower().strip(),
            strength=strength,
            confidence=confidence,
            mechanism=mechanism,
            is_bidirectional=bidirectional
        )
        
        # Add to forward graph
        if relation.cause not in self._causal_graph:
            self._causal_graph[relation.cause] = []
        self._causal_graph[relation.cause].append(relation)
        
        # Add to reverse graph
        if relation.effect not in self._reverse_graph:
            self._reverse_graph[relation.effect] = []
        self._reverse_graph[relation.effect].append(relation)
        
        # Handle bidirectional
        if bidirectional:
            reverse = CausalRelation(
                cause=relation.effect,
                effect=relation.cause,
                strength=strength * 0.8,
                confidence=confidence,
                mechanism=f"Reverse of: {mechanism}"
            )
            if relation.effect not in self._causal_graph:
                self._causal_graph[relation.effect] = []
            self._causal_graph[relation.effect].append(reverse)
        
        logger.debug(f"Causal: {relation}")
        return relation

    def what_causes(self, effect: str, max_depth: int = 3) -> CausalQuery:
        """
        Find causes of an observed effect.
        
        Args:
            effect: The observed event/variable
            max_depth: How deep to trace causal chains
            
        Returns:
            CausalQuery with causes and causal chain
        """
        effect_lower = effect.lower().strip()
        
        direct_causes = self._reverse_graph.get(effect_lower, [])
        
        if not direct_causes:
            # Try fuzzy matching
            for key in self._reverse_graph:
                if effect_lower in key or key in effect_lower:
                    direct_causes = self._reverse_graph[key]
                    break
        
        if not direct_causes:
            return CausalQuery(
                query_type="what_causes",
                query=effect,
                answer=f"No known causes found for '{effect}' in knowledge base.",
                causal_chain=[],
                confidence=0.0
            )
        
        # Sort by causal strength
        direct_causes.sort(key=lambda r: r.strength * r.confidence, reverse=True)
        
        primary_cause = direct_causes[0]
        causal_chain = self._trace_causal_chain(primary_cause.cause, effect_lower, max_depth)
        
        answer = f"'{effect}' is primarily caused by '{primary_cause.cause}' (strength={primary_cause.strength:.2f})"
        if primary_cause.mechanism:
            answer += f". Mechanism: {primary_cause.mechanism}"
        
        alternative = [r.cause for r in direct_causes[1:4]]
        
        return CausalQuery(
            query_type="what_causes",
            query=effect,
            answer=answer,
            causal_chain=causal_chain,
            confidence=primary_cause.confidence,
            alternative_causes=alternative
        )

    def what_effects(self, cause: str, max_depth: int = 3) -> CausalQuery:
        """
        Find effects of a cause.
        
        Args:
            cause: The causing event/variable
            max_depth: How deep to trace effect chains
            
        Returns:
            CausalQuery with effects and causal chain
        """
        cause_lower = cause.lower().strip()
        
        direct_effects = self._causal_graph.get(cause_lower, [])
        
        if not direct_effects:
            # Try fuzzy matching
            for key in self._causal_graph:
                if cause_lower in key or key in cause_lower:
                    direct_effects = self._causal_graph[key]
                    break
        
        if not direct_effects:
            return CausalQuery(
                query_type="what_effects",
                query=cause,
                answer=f"No known effects found for '{cause}' in knowledge base.",
                causal_chain=[],
                confidence=0.0
            )
        
        direct_effects.sort(key=lambda r: r.strength * r.confidence, reverse=True)
        
        effects_str = ", ".join(r.effect for r in direct_effects[:3])
        answer = f"'{cause}' causes: {effects_str}"
        
        primary_chain = self._trace_causal_chain(cause_lower, direct_effects[0].effect, max_depth)
        
        return CausalQuery(
            query_type="what_effects",
            query=cause,
            answer=answer,
            causal_chain=primary_chain,
            confidence=sum(r.confidence for r in direct_effects) / len(direct_effects)
        )

    def counterfactual(self, event: str, intervention: str) -> CausalQuery:
        """
        Counterfactual reasoning: What would happen if X were different?
        
        Example: "What if there was no data? (effect on machine learning)"
        
        Args:
            event: The event we're asking about
            intervention: The hypothetical change
            
        Returns:
            CausalQuery with counterfactual analysis
        """
        # Find what the event causes
        effects_query = self.what_effects(event)
        
        if not effects_query.causal_chain:
            answer = f"Without '{intervention}', it's unclear what would change regarding '{event}'."
            confidence = 0.3
        else:
            answer = (f"Counterfactual: If '{intervention}', "
                     f"then '{event}' would be affected, "
                     f"which would impact: {', '.join(effects_query.causal_chain[:3])}")
            confidence = 0.6
        
        return CausalQuery(
            query_type="counterfactual",
            query=f"What if {intervention}?",
            answer=answer,
            causal_chain=effects_query.causal_chain,
            confidence=confidence
        )

    def _trace_causal_chain(self, start: str, end: str, max_depth: int) -> List[str]:
        """Trace a causal chain from start to end."""
        if start == end:
            return [start]
        
        # BFS to find causal path
        from collections import deque
        queue = deque([[start]])
        visited = {start}
        
        while queue:
            path = queue.popleft()
            current = path[-1]
            
            if len(path) > max_depth + 1:
                break
            
            effects = self._causal_graph.get(current, [])
            for relation in effects:
                if relation.effect not in visited:
                    new_path = path + [relation.effect]
                    if relation.effect == end:
                        return new_path
                    visited.add(relation.effect)
                    queue.append(new_path)
        
        return [start, end] if start != end else [start]

    def find_causal_path(self, cause: str, effect: str) -> Optional[List[str]]:
        """Find if there's a causal path from cause to effect."""
        chain = self._trace_causal_chain(cause.lower(), effect.lower(), max_depth=5)
        return chain if len(chain) > 1 else None

    def get_stats(self) -> Dict:
        """Get causal reasoner statistics."""
        total_relations = sum(len(v) for v in self._causal_graph.values())
        return {
            "unique_causes": len(self._causal_graph),
            "unique_effects": len(self._reverse_graph),
            "total_relations": total_relations,
        }
