"""
Self-Reflection Module
========================
Meta-cognitive module that evaluates and improves the AI's own responses.

The Critic-Improve loop:
  1. Generate initial answer
  2. Critic evaluates: quality, accuracy, completeness, clarity
  3. Identify gaps and improvements
  4. Generate improved answer
  5. Repeat until quality threshold met or max iterations reached

Inspired by:
- Constitutional AI (Anthropic)
- Self-Refine (Madaan et al.)
- Reflexion (Shinn et al.)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CritiqueResult:
    """Result of the critic's evaluation."""
    score: float                    # 0.0-1.0 quality score
    strengths: List[str]
    weaknesses: List[str]
    missing_elements: List[str]
    improvement_suggestions: List[str]
    should_improve: bool

    @property
    def overall_quality(self) -> str:
        """Return quality level as string."""
        if self.score >= 0.9:
            return "excellent"
        elif self.score >= 0.7:
            return "good"
        elif self.score >= 0.5:
            return "fair"
        else:
            return "poor"


@dataclass
class ReflectionResult:
    """Final result after self-reflection."""
    original_answer: str
    final_answer: str
    iterations: int
    critique_history: List[CritiqueResult] = field(default_factory=list)
    improvement_history: List[str] = field(default_factory=list)
    final_quality_score: float = 0.0
    was_improved: bool = False


class SelfReflectionModule:
    """
    Self-Reflection and Meta-Cognitive Module
    
    Implements the critic-improve feedback loop:
    
    Query → Answer₀ → Critique₀ → Answer₁ → Critique₁ → ... → Final Answer
    
    The critic evaluates:
    - Factual accuracy
    - Completeness (are all aspects addressed?)
    - Clarity (is it understandable?)
    - Relevance (does it answer the actual question?)
    - Logical consistency (no contradictions?)
    
    Features:
    - LLM-powered critique (when API available)
    - Rule-based critique fallback
    - Configurable max iterations
    - Quality threshold stopping
    """

    CRITIC_PROMPT = """You are a critical evaluator. Assess the following answer to the given question.

Question: {query}

Answer to evaluate:
{answer}

Provide your evaluation in this exact format:
SCORE: [0.0-1.0]
STRENGTHS:
- [strength 1]
- [strength 2]
WEAKNESSES:
- [weakness 1]
- [weakness 2]
MISSING:
- [missing element 1]
SUGGESTIONS:
- [improvement suggestion 1]
SHOULD_IMPROVE: [YES/NO]

Be specific and constructive in your evaluation."""

    IMPROVEMENT_PROMPT = """You are an expert at improving answers. 

Original Question: {query}

Previous Answer:
{answer}

Critique and Suggestions:
{critique}

Please provide an improved answer that addresses all the weaknesses and incorporates the suggestions. 
Make it more accurate, complete, and clear while keeping it concise."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_iterations = self.config.get("max_iterations", 3)
        self.quality_threshold = self.config.get("improvement_threshold", 0.8)
        self._llm = None
        logger.info(f"🔮 Self-Reflection Module initialized (max_iter={self.max_iterations})")

    def _get_llm(self):
        """Lazy-initialize LLM client."""
        if self._llm is None:
            try:
                from openai import OpenAI
                import os
                self._llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self._llm = "fallback"
        return self._llm

    def reflect_and_improve(
        self,
        query: str,
        initial_answer: str,
        context: Dict = None
    ) -> str:
        """
        Main self-reflection pipeline.
        
        Args:
            query: Original question/prompt
            initial_answer: First-pass answer to evaluate
            context: Optional memory context
            
        Returns:
            str: Final improved answer
        """
        logger.info(f"🔮 Starting self-reflection on answer...")
        
        result = self._reflection_loop(query, initial_answer, context)
        
        if result.was_improved:
            logger.info(f"✨ Answer improved in {result.iterations} iteration(s). "
                       f"Quality: {result.final_quality_score:.2f}")
        else:
            logger.info(f"Answer quality acceptable ({result.final_quality_score:.2f}), no improvement needed")
        
        return result.final_answer

    def full_reflect(
        self,
        query: str,
        initial_answer: str,
        context: Dict = None
    ) -> ReflectionResult:
        """
        Full reflection returning detailed result with history.
        
        Returns ReflectionResult with complete critique and improvement history.
        """
        return self._reflection_loop(query, initial_answer, context)

    def _reflection_loop(
        self,
        query: str,
        initial_answer: str,
        context: Dict = None
    ) -> ReflectionResult:
        """Core reflection loop."""
        current_answer = initial_answer
        critique_history = []
        improvement_history = [initial_answer]
        was_improved = False
        
        for iteration in range(self.max_iterations):
            # Step 1: Critique current answer
            critique = self._critique(query, current_answer)
            critique_history.append(critique)
            
            logger.debug(f"Iteration {iteration+1}: Quality score = {critique.score:.2f}")
            
            # Step 2: Check if quality threshold is met
            if critique.score >= self.quality_threshold or not critique.should_improve:
                logger.debug(f"Quality threshold met at iteration {iteration+1}")
                break
            
            # Step 3: Improve the answer
            improved_answer = self._improve(query, current_answer, critique)
            
            if improved_answer and improved_answer != current_answer:
                current_answer = improved_answer
                improvement_history.append(current_answer)
                was_improved = True
            else:
                logger.debug("No significant improvement found, stopping reflection")
                break
        
        final_score = critique_history[-1].score if critique_history else 0.7
        
        return ReflectionResult(
            original_answer=initial_answer,
            final_answer=current_answer,
            iterations=len(critique_history),
            critique_history=critique_history,
            improvement_history=improvement_history,
            final_quality_score=final_score,
            was_improved=was_improved
        )

    def _critique(self, query: str, answer: str) -> CritiqueResult:
        """
        Evaluate the quality of an answer.
        Uses LLM if available, otherwise rule-based evaluation.
        """
        llm = self._get_llm()
        
        if llm != "fallback":
            return self._llm_critique(query, answer)
        else:
            return self._rule_based_critique(query, answer)

    def _llm_critique(self, query: str, answer: str) -> CritiqueResult:
        """LLM-powered critique."""
        llm = self._get_llm()
        
        try:
            prompt = self.CRITIC_PROMPT.format(query=query, answer=answer)
            
            response = llm.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a precise and constructive evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            critique_text = response.choices[0].message.content
            return self._parse_critique(critique_text)
            
        except Exception as e:
            logger.error(f"LLM critique failed: {e}")
            return self._rule_based_critique(query, answer)

    def _rule_based_critique(self, query: str, answer: str) -> CritiqueResult:
        """Rule-based critique when LLM is not available."""
        strengths = []
        weaknesses = []
        missing = []
        suggestions = []
        
        # Check answer length
        word_count = len(answer.split())
        if word_count > 50:
            strengths.append("Sufficiently detailed response")
        elif word_count < 10:
            weaknesses.append("Response is too brief")
            suggestions.append("Provide more detail and explanation")
        
        # Check if answer directly addresses the query
        query_keywords = set(query.lower().split()) - {"what", "how", "why", "is", "are", "the", "a", "an"}
        answer_keywords = set(answer.lower().split())
        overlap = query_keywords & answer_keywords
        
        if len(overlap) > 0:
            strengths.append(f"Addresses key query terms: {', '.join(list(overlap)[:3])}")
        else:
            weaknesses.append("May not directly address the question")
            missing.append("Direct answer to the question")
        
        # Check for structure
        if "\n" in answer or ". " in answer:
            strengths.append("Well-structured response")
        
        # Estimate quality score
        score = 0.5
        score += len(strengths) * 0.1
        score -= len(weaknesses) * 0.1
        score = max(0.0, min(1.0, score))
        
        should_improve = score < self.quality_threshold and bool(suggestions)
        
        return CritiqueResult(
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            missing_elements=missing,
            improvement_suggestions=suggestions,
            should_improve=should_improve
        )

    def _improve(self, query: str, answer: str, critique: CritiqueResult) -> str:
        """Generate improved answer based on critique."""
        llm = self._get_llm()
        
        if llm != "fallback":
            return self._llm_improve(query, answer, critique)
        else:
            return self._rule_based_improve(query, answer, critique)

    def _llm_improve(self, query: str, answer: str, critique: CritiqueResult) -> str:
        """LLM-powered improvement."""
        llm = self._get_llm()
        
        try:
            critique_summary = f"""
Weaknesses: {', '.join(critique.weaknesses)}
Missing: {', '.join(critique.missing_elements)}
Suggestions: {', '.join(critique.improvement_suggestions)}
"""
            
            prompt = self.IMPROVEMENT_PROMPT.format(
                query=query,
                answer=answer,
                critique=critique_summary
            )
            
            response = llm.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You improve answers based on constructive feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM improvement failed: {e}")
            return self._rule_based_improve(query, answer, critique)

    def _rule_based_improve(self, query: str, answer: str, critique: CritiqueResult) -> str:
        """Rule-based improvement."""
        if not critique.improvement_suggestions:
            return answer
        
        # Append suggestions as additional context
        improved = answer
        if critique.missing_elements:
            improved += f"\n\nAdditionally: {', '.join(critique.missing_elements[:2])} should be considered."
        
        return improved

    def _parse_critique(self, critique_text: str) -> CritiqueResult:
        """Parse LLM critique response into structured format."""
        lines = critique_text.strip().split("\n")
        
        score = 0.7
        strengths = []
        weaknesses = []
        missing = []
        suggestions = []
        should_improve = False
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif "STRENGTHS:" in line:
                current_section = "strengths"
            elif "WEAKNESSES:" in line:
                current_section = "weaknesses"
            elif "MISSING:" in line:
                current_section = "missing"
            elif "SUGGESTIONS:" in line:
                current_section = "suggestions"
            elif "SHOULD_IMPROVE:" in line:
                should_improve = "YES" in line.upper()
                current_section = None
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "strengths":
                    strengths.append(item)
                elif current_section == "weaknesses":
                    weaknesses.append(item)
                elif current_section == "missing":
                    missing.append(item)
                elif current_section == "suggestions":
                    suggestions.append(item)
        
        return CritiqueResult(
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            missing_elements=missing,
            improvement_suggestions=suggestions,
            should_improve=should_improve
        )
