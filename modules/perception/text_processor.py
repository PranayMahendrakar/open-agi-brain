"""
Perception Engine - Text Processor
====================================
Processes natural language text input using NLP models.
Handles tokenization, entity extraction, intent detection, and semantic understanding.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextPerception:
    """Structured output from text perception."""
    raw_text: str
    tokens: List[str]
    entities: List[Dict[str, str]]
    intent: str
    sentiment: float          # -1.0 (negative) to 1.0 (positive)
    language: str
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None


class TextProcessor:
    """
    Text Perception Module
    
    Processes raw text input and extracts:
    - Named entities (people, places, organizations)
    - Intent classification
    - Sentiment analysis
    - Semantic embeddings for memory storage
    - Language detection
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._embedder = None
        self._nlp = None
        logger.info("📝 Text Processor initialized")

    def _get_embedder(self):
        """Lazy load sentence transformer embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
                self._embedder = SentenceTransformer(model_name)
                logger.info(f"Loaded embedder: {model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Embeddings disabled.")
        return self._embedder

    def _get_nlp(self):
        """Lazy load spaCy NLP model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except (ImportError, OSError):
                logger.warning("spaCy model not available. Basic NLP only.")
        return self._nlp

    def process(self, text: str) -> TextPerception:
        """
        Process raw text through the full NLP pipeline.
        
        Args:
            text: Raw input text string
            
        Returns:
            TextPerception: Structured perception output
        """
        logger.debug(f"Processing text: {text[:100]}...")

        # Basic tokenization
        tokens = self._tokenize(text)

        # Entity extraction
        entities = self._extract_entities(text)

        # Intent detection
        intent = self._detect_intent(text)

        # Sentiment analysis
        sentiment = self._analyze_sentiment(text)

        # Language detection
        language = self._detect_language(text)

        # Generate embedding
        embedding = self._embed(text)

        perception = TextPerception(
            raw_text=text,
            tokens=tokens,
            entities=entities,
            intent=intent,
            sentiment=sentiment,
            language=language,
            embedding=embedding,
        )

        logger.debug(f"Text processed. Intent: {intent}, Sentiment: {sentiment:.2f}")
        return perception

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        nlp = self._get_nlp()
        if nlp:
            doc = nlp(text)
            return [token.text for token in doc if not token.is_space]
        return text.split()

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        nlp = self._get_nlp()
        if nlp:
            doc = nlp(text)
            return [
                {"text": ent.text, "label": ent.label_, "description": ent.label_}
                for ent in doc.ents
            ]
        return []

    def _detect_intent(self, text: str) -> str:
        """
        Simple rule-based intent detection.
        
        Intents: question, command, statement, greeting, farewell
        """
        text_lower = text.lower().strip()

        # Question detection
        question_words = ["what", "who", "where", "when", "why", "how", "is", "are", "can", "does"]
        if text_lower.endswith("?") or any(text_lower.startswith(w) for w in question_words):
            return "question"

        # Command detection
        command_words = ["do", "make", "create", "build", "run", "execute", "find", "search", "get"]
        if any(text_lower.startswith(w) for w in command_words):
            return "command"

        # Greeting detection
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        if any(text_lower.startswith(g) for g in greetings):
            return "greeting"

        # Farewell
        farewells = ["bye", "goodbye", "see you", "farewell", "ciao"]
        if any(text_lower.startswith(f) for f in farewells):
            return "farewell"

        return "statement"

    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple keyword-based sentiment analysis.
        Returns value between -1.0 (negative) and 1.0 (positive).
        """
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "love", "best", "happy", "positive", "awesome"}
        negative_words = {"bad", "terrible", "awful", "horrible", "hate", "worst", "sad", "negative", "poor", "wrong"}

        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count

        if total == 0:
            return 0.0  # Neutral

        return (pos_count - neg_count) / total

    def _detect_language(self, text: str) -> str:
        """Basic language detection (defaults to English)."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate semantic embedding for the text."""
        embedder = self._get_embedder()
        if embedder:
            return embedder.encode(text).tolist()
        return None

    def batch_process(self, texts: List[str]) -> List[TextPerception]:
        """Process multiple texts in batch."""
        return [self.process(text) for text in texts]
