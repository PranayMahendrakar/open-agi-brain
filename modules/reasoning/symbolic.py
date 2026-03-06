"""
Reasoning Engine - Symbolic Reasoning
========================================
Rule-based logical reasoning using symbolic AI techniques.
Applies formal logic rules to draw conclusions from facts.

Implements:
- Forward chaining (fact → rule → conclusion)
- Backward chaining (goal → rule → required facts)
- Knowledge base management
- First-order logic patterns

Analogous to: Deliberate, rule-following reasoning (System 2 thinking)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Rule:
    """A logical inference rule: IF conditions THEN conclusion."""
    rule_id: str
    conditions: List[str]    # List of condition strings
    conclusion: str          # What can be concluded
    confidence: float = 1.0  # Confidence in this rule
    description: str = ""


@dataclass
class Fact:
    """A logical fact in the knowledge base."""
    fact_id: str
    statement: str
    confidence: float = 1.0
    source: str = "user"


@dataclass
class InferenceResult:
    """Result of symbolic reasoning."""
    query: str
    conclusion: Optional[str]
    supporting_facts: List[str]
    applied_rules: List[str]
    confidence: float
    is_provable: bool


class SymbolicReasoner:
    """
    Symbolic/Rule-Based Reasoning Engine
    
    Uses formal logic to:
    1. Maintain a knowledge base of facts
    2. Apply inference rules to derive new conclusions
    3. Check logical consistency
    4. Answer yes/no questions via proof
    
    Methods:
    - Forward chaining: Start from facts, apply rules, reach conclusions
    - Backward chaining: Start from goal, work backwards to find supporting facts
    
    Analogous to: Expert systems, Prolog-style reasoning
    """

    # Built-in common sense rules
    DEFAULT_RULES = [
        Rule("r001", ["X is_a mammal"], "X is_a animal", 1.0, "All mammals are animals"),
        Rule("r002", ["X is_a animal", "X can fly"], "X is_a bird", 0.7, "Flying animals may be birds"),
        Rule("r003", ["X is_a human"], "X is_a mammal", 1.0, "Humans are mammals"),
        Rule("r004", ["X is_a human"], "X has intelligence", 0.95, "Humans have intelligence"),
        Rule("r005", ["X has_property artificial", "X has intelligence"], "X is_a AI", 0.9, "Artificial intelligence"),
        Rule("r006", ["X is_a programming_language", "X is high_level"], "X is easy_to_learn", 0.7, "High-level languages are easier"),
    ]

    def __init__(self):
        self._facts: Dict[str, Fact] = {}
        self._rules: Dict[str, Rule] = {}
        self._derived_facts: Set[str] = set()
        
        # Load default rules
        for rule in self.DEFAULT_RULES:
            self._rules[rule.rule_id] = rule
        
        logger.info(f"🔷 Symbolic Reasoner initialized ({len(self._rules)} default rules)")

    def add_fact(self, statement: str, confidence: float = 1.0, source: str = "user") -> str:
        """
        Add a fact to the knowledge base.
        
        Args:
            statement: Fact as string (e.g., "Python is_a programming_language")
            confidence: Confidence level
            source: Where this fact came from
            
        Returns:
            str: Fact ID
        """
        fact_id = f"f{len(self._facts):04d}"
        self._facts[fact_id] = Fact(
            fact_id=fact_id,
            statement=statement.strip().lower(),
            confidence=confidence,
            source=source
        )
        logger.debug(f"Fact added: '{statement}'")
        return fact_id

    def add_rule(self, conditions: List[str], conclusion: str, confidence: float = 1.0, description: str = "") -> str:
        """
        Add a custom inference rule.
        
        Args:
            conditions: List of condition patterns
            conclusion: What can be concluded
            confidence: Rule confidence
            description: Human-readable description
            
        Returns:
            str: Rule ID
        """
        rule_id = f"r{len(self._rules):04d}"
        self._rules[rule_id] = Rule(
            rule_id=rule_id,
            conditions=[c.strip().lower() for c in conditions],
            conclusion=conclusion.strip().lower(),
            confidence=confidence,
            description=description
        )
        logger.debug(f"Rule added: IF {conditions} THEN '{conclusion}'")
        return rule_id

    def forward_chain(self, max_iterations: int = 10) -> List[str]:
        """
        Forward chaining: Apply all rules to derive new facts.
        
        Starts from known facts and repeatedly applies rules
        until no new facts can be derived.
        
        Returns:
            List of newly derived fact statements
        """
        known_facts = {f.statement for f in self._facts.values()}
        derived = set()
        
        for _ in range(max_iterations):
            new_facts_this_round = set()
            
            for rule in self._rules.values():
                # Check if all conditions are satisfied
                if self._conditions_satisfied(rule.conditions, known_facts | derived):
                    conclusion = rule.conclusion
                    if conclusion not in known_facts and conclusion not in derived:
                        derived.add(conclusion)
                        new_facts_this_round.add(conclusion)
                        logger.debug(f"Derived: '{conclusion}' via rule '{rule.rule_id}'")
            
            if not new_facts_this_round:
                break  # Fixed point reached
        
        self._derived_facts.update(derived)
        return list(derived)

    def query(self, goal: str) -> InferenceResult:
        """
        Query whether a statement can be proven true.
        
        Uses both direct fact lookup and backward chaining.
        
        Args:
            goal: Statement to prove (e.g., "Python is_a animal")
            
        Returns:
            InferenceResult with proof status and supporting evidence
        """
        goal_lower = goal.strip().lower()
        
        # First, run forward chaining to derive all possible facts
        self.forward_chain()
        
        # Check direct facts
        all_facts = {f.statement for f in self._facts.values()} | self._derived_facts
        
        if goal_lower in all_facts:
            supporting = self._find_supporting_facts(goal_lower)
            return InferenceResult(
                query=goal,
                conclusion=goal,
                supporting_facts=supporting,
                applied_rules=[],
                confidence=0.9,
                is_provable=True
            )
        
        # Try backward chaining
        proof = self._backward_chain(goal_lower, all_facts, depth=0)
        
        return InferenceResult(
            query=goal,
            conclusion=goal_lower if proof else None,
            supporting_facts=list(all_facts)[:5] if proof else [],
            applied_rules=[r.rule_id for r in self._rules.values() if proof][:3],
            confidence=0.7 if proof else 0.0,
            is_provable=proof
        )

    def _conditions_satisfied(self, conditions: List[str], known_facts: Set[str]) -> bool:
        """Check if all rule conditions are met by known facts."""
        for condition in conditions:
            # Simple pattern matching (could be enhanced with unification)
            if not any(self._matches(condition, fact) for fact in known_facts):
                return False
        return True

    def _matches(self, pattern: str, fact: str) -> bool:
        """
        Check if a pattern matches a fact.
        Supports simple variable substitution (X as wildcard).
        """
        if "X" not in pattern:
            return pattern == fact
        
        # Replace X with regex-like wildcard
        # Simple approach: check structural similarity
        pattern_parts = pattern.split()
        fact_parts = fact.split()
        
        if len(pattern_parts) != len(fact_parts):
            return False
        
        for pp, fp in zip(pattern_parts, fact_parts):
            if pp != "X" and pp != fp:
                return False
        
        return True

    def _backward_chain(self, goal: str, known_facts: Set[str], depth: int, max_depth: int = 5) -> bool:
        """
        Backward chaining: Work from goal to supporting facts.
        Returns True if goal can be proven.
        """
        if depth > max_depth:
            return False
        
        if goal in known_facts:
            return True
        
        # Find rules whose conclusion matches the goal
        for rule in self._rules.values():
            if self._matches(rule.conclusion, goal) or rule.conclusion == goal:
                # Try to prove all conditions of this rule
                all_conditions_met = all(
                    self._backward_chain(cond, known_facts, depth + 1, max_depth)
                    for cond in rule.conditions
                    if "X" not in cond
                )
                if all_conditions_met:
                    return True
        
        return False

    def _find_supporting_facts(self, statement: str) -> List[str]:
        """Find facts that support a given statement."""
        facts = [f.statement for f in self._facts.values()]
        return [f for f in facts if any(word in statement for word in f.split()[:2])][:5]

    def explain(self, goal: str) -> str:
        """Generate a human-readable explanation of the reasoning."""
        result = self.query(goal)
        
        if result.is_provable:
            explanation = f"'{goal}' is TRUE.
"
            if result.supporting_facts:
                explanation += f"Supporting facts:
"
                for fact in result.supporting_facts[:3]:
                    explanation += f"  - {fact}\n"
        else:
            explanation = f"'{goal}' cannot be proven with current knowledge base."
        
        return explanation

    def get_all_facts(self) -> List[str]:
        """Get all facts including derived ones."""
        self.forward_chain()
        facts = [f.statement for f in self._facts.values()]
        facts += list(self._derived_facts)
        return facts

    def get_stats(self) -> Dict:
        """Get reasoner statistics."""
        return {
            "user_facts": len(self._facts),
            "derived_facts": len(self._derived_facts),
            "total_rules": len(self._rules),
            "default_rules": len(self.DEFAULT_RULES),
        }
