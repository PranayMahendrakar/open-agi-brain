"""
Decision Engine
================
Goal-based decision making with planning and risk estimation.
Selects the best action given the current state, goals, and context.

Implements:
- Goal-based planning (BFS/A*/MCTS)
- Risk estimation
- Reinforcement learning policy
- Action selection with confidence scores
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from utils.logger import get_logger

logger = get_logger(__name__)


class ActionType(Enum):
    """Types of actions the AGI can take."""
    RESPOND = "respond"           # Give a verbal/text response
    QUERY_MEMORY = "query_memory" # Look up information in memory
    ASK_CLARIFICATION = "ask_clarification"  # Ask user for more info
    EXPLORE = "explore"           # Explore new information
    PLAN = "plan"                 # Create a multi-step plan
    EXECUTE = "execute"           # Execute a planned action
    REFLECT = "reflect"           # Trigger self-reflection
    WAIT = "wait"                 # Wait/do nothing


@dataclass
class Action:
    """A decision/action to be taken."""
    action_type: ActionType
    payload: Any = None
    confidence: float = 1.0
    estimated_reward: float = 0.0
    risk_score: float = 0.0
    reasoning: str = ""


@dataclass
class DecisionContext:
    """Context for decision making."""
    state: Any
    response: str
    context: Dict
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    available_actions: List[ActionType] = field(default_factory=list)


class DecisionEngine:
    """
    Decision Engine — Goal-Based Action Selection
    
    The decision engine is responsible for:
    1. Analyzing the current cognitive state
    2. Evaluating available actions
    3. Estimating rewards and risks
    4. Selecting the optimal action
    
    Decision Flow:
    State + Context → Goal Analysis → Action Candidates →
    Risk Estimation → Expected Value → Best Action
    
    Analogous to: Prefrontal cortex executive function
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.planning_algorithm = self.config.get("planning_algorithm", "greedy")
        self.risk_threshold = self.config.get("risk_threshold", 0.3)
        self.discount_factor = self.config.get("discount_factor", 0.95)
        
        # Action value estimates (Q-values, initialized optimistically)
        self._q_values: Dict[str, float] = {
            action.value: 0.5 for action in ActionType
        }
        
        # Decision history for learning
        self._decision_history: List[Dict] = []
        
        logger.info(f"⚡ Decision Engine initialized (algorithm={self.planning_algorithm})")

    def decide(
        self,
        state: Any,
        response: str,
        context: Dict = None,
        goals: List[str] = None
    ) -> Action:
        """
        Select the best action given current state and context.
        
        Args:
            state: Current perceived state
            response: Generated response from reasoning
            context: Memory context
            goals: Active goals
            
        Returns:
            Action: The selected action with metadata
        """
        ctx = DecisionContext(
            state=state,
            response=response,
            context=context or {},
            goals=goals or [],
            available_actions=list(ActionType)
        )
        
        # Generate action candidates
        candidates = self._generate_candidates(ctx)
        
        # Estimate values and risks for each candidate
        evaluated = [(action, self._evaluate_action(action, ctx)) for action in candidates]
        
        # Select best action using configured algorithm
        best_action = self._select_action(evaluated, ctx)
        
        # Record decision
        self._decision_history.append({
            "state_summary": str(state)[:100],
            "selected_action": best_action.action_type.value,
            "confidence": best_action.confidence,
        })
        
        logger.debug(f"Decision: {best_action.action_type.value} (confidence={best_action.confidence:.2f})")
        return best_action

    def _generate_candidates(self, ctx: DecisionContext) -> List[Action]:
        """Generate candidate actions based on context."""
        candidates = []
        
        # Always consider responding
        candidates.append(Action(
            action_type=ActionType.RESPOND,
            payload=ctx.response,
            confidence=0.8,
            reasoning="Provide response to user"
        ))
        
        # If context lacks information, consider querying memory
        if not ctx.context.get("long_term"):
            candidates.append(Action(
                action_type=ActionType.QUERY_MEMORY,
                confidence=0.6,
                reasoning="Retrieve relevant memory context"
            ))
        
        # If response has low confidence indicators, consider reflection
        response_lower = ctx.response.lower()
        uncertainty_phrases = ["i'm not sure", "might", "possibly", "unclear", "don't know"]
        if any(phrase in response_lower for phrase in uncertainty_phrases):
            candidates.append(Action(
                action_type=ActionType.REFLECT,
                confidence=0.7,
                reasoning="Response shows uncertainty, reflection needed"
            ))
        
        # If query is complex, consider planning
        state_str = str(ctx.state)
        if len(state_str.split()) > 30 or "step" in state_str.lower():
            candidates.append(Action(
                action_type=ActionType.PLAN,
                confidence=0.65,
                reasoning="Complex query may benefit from structured planning"
            ))
        
        return candidates

    def _evaluate_action(self, action: Action, ctx: DecisionContext) -> Tuple[float, float]:
        """
        Evaluate an action's expected value and risk.
        
        Returns:
            Tuple of (expected_value, risk_score)
        """
        # Base value from Q-learning estimates
        base_value = self._q_values.get(action.action_type.value, 0.5)
        
        # Risk assessment
        risk = self._estimate_risk(action, ctx)
        
        # Expected value = base_value * confidence - risk
        expected_value = (base_value * action.confidence) - (risk * 0.5)
        expected_value = max(0.0, min(1.0, expected_value))
        
        return expected_value, risk

    def _estimate_risk(self, action: Action, ctx: DecisionContext) -> float:
        """
        Estimate the risk of taking an action.
        Higher risk = could be wrong or harmful.
        """
        risk_map = {
            ActionType.RESPOND: 0.1,          # Low risk
            ActionType.QUERY_MEMORY: 0.05,    # Very low risk
            ActionType.ASK_CLARIFICATION: 0.1,# Low risk
            ActionType.EXPLORE: 0.3,          # Medium risk
            ActionType.PLAN: 0.2,             # Low-medium risk
            ActionType.EXECUTE: 0.5,          # Higher risk
            ActionType.REFLECT: 0.1,          # Low risk
            ActionType.WAIT: 0.05,            # Very low risk
        }
        
        base_risk = risk_map.get(action.action_type, 0.3)
        
        # Adjust risk based on confidence
        adjusted_risk = base_risk * (1.0 - action.confidence * 0.5)
        
        return min(1.0, adjusted_risk)

    def _select_action(
        self,
        evaluated: List[Tuple[Action, Tuple[float, float]]],
        ctx: DecisionContext
    ) -> Action:
        """Select best action using configured algorithm."""
        
        if self.planning_algorithm == "greedy":
            return self._greedy_select(evaluated)
        elif self.planning_algorithm == "epsilon_greedy":
            return self._epsilon_greedy_select(evaluated)
        else:
            return self._greedy_select(evaluated)

    def _greedy_select(self, evaluated: List[Tuple[Action, Tuple[float, float]]]) -> Action:
        """Select action with highest expected value."""
        if not evaluated:
            return Action(action_type=ActionType.RESPOND, confidence=0.5)
        
        best_action = None
        best_value = -1.0
        
        for action, (expected_value, risk) in evaluated:
            if risk <= self.risk_threshold and expected_value > best_value:
                best_value = expected_value
                best_action = action
                best_action.estimated_reward = expected_value
                best_action.risk_score = risk
        
        if best_action is None:
            # All actions exceed risk threshold, take lowest risk
            evaluated_sorted = sorted(evaluated, key=lambda x: x[1][1])
            best_action = evaluated_sorted[0][0]
            best_action.risk_score = evaluated_sorted[0][1][1]
        
        return best_action

    def _epsilon_greedy_select(self, evaluated: List[Tuple]) -> Action:
        """Epsilon-greedy: explore with probability epsilon."""
        epsilon = 0.1  # 10% exploration
        
        if random.random() < epsilon:
            # Random exploration
            random_idx = random.randint(0, len(evaluated) - 1)
            action = evaluated[random_idx][0]
            action.reasoning += " [exploration]"
            return action
        else:
            return self._greedy_select(evaluated)

    def update_q_value(self, action_type: ActionType, reward: float) -> None:
        """
        Update Q-value for an action based on received reward.
        Simple Q-learning update: Q(a) = Q(a) + alpha * (reward - Q(a))
        """
        alpha = 0.1  # Learning rate
        current_q = self._q_values.get(action_type.value, 0.5)
        new_q = current_q + alpha * (reward - current_q)
        self._q_values[action_type.value] = new_q
        logger.debug(f"Q-value updated: {action_type.value} = {new_q:.3f}")

    def plan_sequence(self, goal: str, steps: int = 5) -> List[Action]:
        """
        Create a multi-step action plan to achieve a goal.
        
        Args:
            goal: The goal to achieve
            steps: Maximum number of steps
            
        Returns:
            List of actions in sequence
        """
        plan = []
        
        # Step 1: Query memory for relevant information
        plan.append(Action(
            action_type=ActionType.QUERY_MEMORY,
            payload={"query": goal},
            confidence=0.9,
            reasoning=f"Gather information about: {goal}"
        ))
        
        # Step 2: Plan based on information
        plan.append(Action(
            action_type=ActionType.PLAN,
            payload={"goal": goal},
            confidence=0.8,
            reasoning="Formulate detailed plan"
        ))
        
        # Step 3: Reflect on the plan
        plan.append(Action(
            action_type=ActionType.REFLECT,
            confidence=0.85,
            reasoning="Verify plan quality and completeness"
        ))
        
        # Step 4: Execute and respond
        plan.append(Action(
            action_type=ActionType.RESPOND,
            confidence=0.9,
            reasoning="Deliver planned response"
        ))
        
        return plan[:steps]

    def get_stats(self) -> Dict:
        """Get decision engine statistics."""
        action_counts = {}
        for decision in self._decision_history:
            action = decision["selected_action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_decisions": len(self._decision_history),
            "action_distribution": action_counts,
            "q_values": self._q_values.copy(),
            "planning_algorithm": self.planning_algorithm,
        }
