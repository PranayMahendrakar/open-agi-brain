"""
Memory System - Semantic Memory (Knowledge Graph)
==================================================
Implements semantic/world knowledge using a knowledge graph structure.
Stores facts, concepts, and relationships between entities.

Analogous to: Human semantic memory — our knowledge of the world,
facts, meanings, and concepts (not tied to specific events).

Implemented using NetworkX for graph operations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class SemanticMemory:
    """
    Semantic Memory — World Knowledge Graph
    
    Stores:
    - Concepts (nodes) with properties
    - Relationships (edges) between concepts
    - Facts as (subject, predicate, object) triples
    
    Features:
    - Graph-based knowledge representation
    - Relationship traversal (find connected concepts)
    - Query by concept or relationship type
    - Import/export knowledge bases
    
    Example:
        memory = SemanticMemory()
        memory.add_fact("Python", "is_a", "programming_language")
        memory.add_fact("Python", "created_by", "Guido van Rossum")
        memory.query("Python")  # Returns all facts about Python
    """

    def __init__(self):
        self._graph = None
        self._nx_available = False
        self._fallback_facts: List[Tuple] = []
        self._init_graph()
        logger.info("🌐 Semantic Memory (Knowledge Graph) initialized")

    def _init_graph(self):
        """Initialize the knowledge graph."""
        try:
            import networkx as nx
            self._graph = nx.DiGraph()
            self._nx_available = True
            logger.debug("NetworkX knowledge graph initialized")
        except ImportError:
            logger.warning("NetworkX not available. Using list-based fallback.")
            self._nx_available = False

    def add_concept(self, concept: str, properties: Dict = None) -> None:
        """
        Add a concept (node) to the knowledge graph.
        
        Args:
            concept: Concept name
            properties: Optional dict of properties
        """
        if self._nx_available:
            self._graph.add_node(concept, **(properties or {}))
        else:
            self._fallback_facts.append((concept, "exists", concept, properties or {}))

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> None:
        """
        Add a fact/relationship to the knowledge graph.
        
        Args:
            subject: Subject entity
            predicate: Relationship type
            obj: Object entity
            confidence: Confidence score (0.0-1.0)
        """
        if self._nx_available:
            # Ensure nodes exist
            if not self._graph.has_node(subject):
                self._graph.add_node(subject)
            if not self._graph.has_node(obj):
                self._graph.add_node(obj)
            
            # Add edge with relationship
            self._graph.add_edge(
                subject, obj,
                predicate=predicate,
                confidence=confidence
            )
        else:
            self._fallback_facts.append((subject, predicate, obj, {"confidence": confidence}))
        
        logger.debug(f"Knowledge: ({subject}) --[{predicate}]--> ({obj})")

    def query(self, concept: str, max_depth: int = 2) -> List[Dict]:
        """
        Query the knowledge graph for all facts about a concept.
        
        Args:
            concept: Concept to query
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of fact dicts with subject, predicate, object, confidence
        """
        if self._nx_available:
            return self._nx_query(concept, max_depth)
        else:
            return self._fallback_query(concept)

    def _nx_query(self, concept: str, max_depth: int) -> List[Dict]:
        """Query using NetworkX graph."""
        facts = []
        
        if not self._graph.has_node(concept):
            # Try partial match
            matching_nodes = [n for n in self._graph.nodes() 
                            if concept.lower() in str(n).lower()]
            if matching_nodes:
                concept = matching_nodes[0]
            else:
                return []
        
        # Get direct relationships (outgoing)
        for neighbor in self._graph.successors(concept):
            edge_data = self._graph.edges[concept, neighbor]
            facts.append({
                "subject": concept,
                "predicate": edge_data.get("predicate", "related_to"),
                "object": neighbor,
                "confidence": edge_data.get("confidence", 1.0),
                "direction": "outgoing"
            })
        
        # Get reverse relationships (incoming)
        for predecessor in self._graph.predecessors(concept):
            edge_data = self._graph.edges[predecessor, concept]
            facts.append({
                "subject": predecessor,
                "predicate": edge_data.get("predicate", "related_to"),
                "object": concept,
                "confidence": edge_data.get("confidence", 1.0),
                "direction": "incoming"
            })
        
        return facts

    def _fallback_query(self, concept: str) -> List[Dict]:
        """Fallback query using list."""
        results = []
        concept_lower = concept.lower()
        
        for subject, predicate, obj, props in self._fallback_facts:
            if concept_lower in subject.lower() or concept_lower in str(obj).lower():
                results.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": props.get("confidence", 1.0),
                })
        
        return results

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the relationship path between two concepts."""
        if not self._nx_available:
            return None
        
        try:
            import networkx as nx
            if self._graph.has_node(source) and self._graph.has_node(target):
                return nx.shortest_path(self._graph, source, target)
        except Exception:
            pass
        return None

    def get_related_concepts(self, concept: str, relationship: str = None) -> List[str]:
        """Get concepts related to a given concept."""
        facts = self.query(concept)
        
        if relationship:
            related = [f["object"] for f in facts 
                      if f["predicate"] == relationship and f["direction"] == "outgoing"]
            related += [f["subject"] for f in facts
                       if f["predicate"] == relationship and f["direction"] == "incoming"]
        else:
            related = [f["object"] for f in facts if f["direction"] == "outgoing"]
            related += [f["subject"] for f in facts if f["direction"] == "incoming"]
        
        return list(set(related))

    def add_knowledge_base(self, facts: List[Tuple[str, str, str]]) -> int:
        """
        Bulk-add facts to the knowledge graph.
        
        Args:
            facts: List of (subject, predicate, object) tuples
            
        Returns:
            int: Number of facts added
        """
        for subject, predicate, obj in facts:
            self.add_fact(subject, predicate, obj)
        
        logger.info(f"Added {len(facts)} facts to semantic memory")
        return len(facts)

    def node_count(self) -> int:
        """Return number of concept nodes."""
        if self._nx_available:
            return self._graph.number_of_nodes()
        return len(set(s for s, _, _, _ in self._fallback_facts))

    def edge_count(self) -> int:
        """Return number of relationships."""
        if self._nx_available:
            return self._graph.number_of_edges()
        return len(self._fallback_facts)

    def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "backend": "networkx" if self._nx_available else "list_fallback",
        }

    def __repr__(self) -> str:
        return f"SemanticMemory(nodes={self.node_count()}, edges={self.edge_count()})"
