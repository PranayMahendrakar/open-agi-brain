"""
Open AGI Brain - End-to-End Demo
===================================
Demonstrates the full cognitive pipeline of the Open AGI Brain Framework.

Run:
    python examples/demo.py

Prerequisites:
    pip install -r requirements.txt
    export OPENAI_API_KEY="your-api-key"  # Optional, works without it
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.orchestrator import CognitiveOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()


def print_banner():
    """Print the Open AGI Brain banner."""
    banner = """
    ██████╗ ██████╗ ███████╗███╗   ██╗     █████╗  ██████╗ ██╗    
   ██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██╔══██╗██╔════╝ ██║    
   ██║   ██║██████╔╝█████╗  ██╔██╗ ██║    ███████║██║  ███╗██║    
   ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ██╔══██║██║   ██║██║    
   ╚██████╔╝██║     ███████╗██║ ╚████║    ██║  ██║╚██████╔╝██║    
    ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝    
                                                                     
    🧠 Open AGI Brain — Artificial Cognitive Architecture
    An Operating System for Intelligence
    """
    console.print(Panel(banner, style="bold blue"))


def demo_basic_conversation():
    """Demo: Basic text conversation through all cognitive modules."""
    console.print("\n[bold yellow]═══ Demo 1: Basic Conversation ═══[/bold yellow]")
    
    brain = CognitiveOrchestrator()
    
    queries = [
        "What is artificial general intelligence?",
        "How does the human brain process information?",
        "What are the key challenges in building AGI systems?",
    ]
    
    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold cyan]Query {i}:[/bold cyan] {query}")
        
        result = brain.process({
            "type": "text",
            "content": query
        })
        
        # Display results in a table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Response", result["response"][:200] + "..." if len(result["response"]) > 200 else result["response"])
        table.add_row("Action", result["action"].action_type.value if result.get("action") else "N/A")
        table.add_row("Confidence", f"{result.get('confidence', 0):.2f}")
        table.add_row("Novelty Score", f"{result.get('novelty_score', 0):.2f}")
        table.add_row("Reasoning Steps", str(len(result.get("reasoning_steps", []))))
        
        console.print(table)


def demo_memory_system():
    """Demo: Memory storage and retrieval."""
    console.print("\n[bold yellow]═══ Demo 2: Memory System ═══[/bold yellow]")
    
    brain = CognitiveOrchestrator()
    
    # Store some knowledge
    console.print("[cyan]Storing knowledge in memory...[/cyan]")
    
    facts = [
        "The capital of France is Paris.",
        "Python is a high-level programming language created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence.",
        "The human brain has approximately 86 billion neurons.",
    ]
    
    for fact in facts:
        brain.process({"type": "text", "content": fact})
    
    console.print(f"[green]✓ Stored {len(facts)} facts in long-term memory[/green]")
    
    # Retrieve memories
    console.print("\n[cyan]Retrieving relevant memories...[/cyan]")
    memories = brain.remember("artificial intelligence and machine learning", top_k=3)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Memory Type", style="cyan")
    table.add_column("Count", style="white")
    
    table.add_row("Short-term", str(len(memories.get("short_term", []))))
    table.add_row("Long-term", str(len(memories.get("long_term", []))))
    table.add_row("Episodic", str(len(memories.get("episodic", []))))
    table.add_row("Semantic", str(len(memories.get("semantic", []))))
    
    console.print(table)
    
    # Show retrieved memories
    if memories.get("long_term"):
        console.print("\n[cyan]Retrieved long-term memories:[/cyan]")
        for i, mem in enumerate(memories["long_term"][:3], 1):
            console.print(f"  {i}. [dim]{mem.get('content', '')[:100]}[/dim]")


def demo_curiosity_system():
    """Demo: Novelty detection and curiosity."""
    console.print("\n[bold yellow]═══ Demo 3: Curiosity System ═══[/bold yellow]")
    
    from modules.curiosity.curiosity_engine import CuriosityEngine
    from modules.memory.long_term import LongTermMemory
    
    ltm = LongTermMemory("./data/demo_chroma")
    curiosity = CuriosityEngine(
        config={"novelty_threshold": 0.3, "exploration_rate": 0.1},
        long_term_memory=ltm
    )
    
    test_inputs = [
        "What is machine learning?",         # Likely novel at first
        "What is machine learning?",         # Same input - low novelty
        "Quantum entanglement in photons",   # Novel scientific topic
        "Recipe for chocolate cake",         # Completely different domain
    ]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Input", style="cyan", max_width=40)
    table.add_column("Novelty Score", style="white")
    table.add_column("Is Novel?", style="white")
    table.add_column("Interest Level", style="white")
    
    for text in test_inputs:
        patterns = curiosity.detect_interesting_patterns(text)
        is_novel = "✅ Yes" if patterns["is_novel"] else "❌ No"
        interest_color = {
            "high": "bold red",
            "medium": "bold yellow",
            "low": "dim"
        }.get(patterns["interest_level"], "white")
        
        table.add_row(
            text[:40],
            f"{patterns['novelty_score']:.3f}",
            is_novel,
            f"[{interest_color}]{patterns['interest_level']}[/{interest_color}]"
        )
    
    console.print(table)
    
    stats = curiosity.get_stats()
    console.print(f"\n[dim]Total intrinsic reward earned: {stats['total_intrinsic_reward']:.3f}[/dim]")


def demo_self_reflection():
    """Demo: Self-reflection and answer improvement."""
    console.print("\n[bold yellow]═══ Demo 4: Self-Reflection Loop ═══[/bold yellow]")
    
    from modules.self_reflection.reflection_module import SelfReflectionModule
    
    reflector = SelfReflectionModule(
        config={"max_iterations": 2, "improvement_threshold": 0.8}
    )
    
    query = "Explain how neural networks learn from data."
    initial_answer = "Neural networks learn by adjusting weights."
    
    console.print(f"[cyan]Query:[/cyan] {query}")
    console.print(f"[yellow]Initial Answer:[/yellow] {initial_answer}")
    console.print("\n[dim]Running self-reflection loop...[/dim]")
    
    result = reflector.full_reflect(
        query=query,
        initial_answer=initial_answer
    )
    
    console.print(f"\n[green]Final Answer:[/green] {result.final_answer[:300]}")
    console.print(f"[dim]Iterations: {result.iterations} | Quality: {result.final_quality_score:.2f} | Improved: {result.was_improved}[/dim]")
    
    if result.critique_history:
        critique = result.critique_history[0]
        console.print(f"\n[dim]Critique - Score: {critique.score:.2f}[/dim]")
        if critique.strengths:
            console.print(f"[dim]  Strengths: {', '.join(critique.strengths[:2])}[/dim]")
        if critique.weaknesses:
            console.print(f"[dim]  Weaknesses: {', '.join(critique.weaknesses[:2])}[/dim]")


def demo_system_status():
    """Demo: System status overview."""
    console.print("\n[bold yellow]═══ System Status ═══[/bold yellow]")
    
    brain = CognitiveOrchestrator()
    status = brain.get_status()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Short-term Memory Size", str(status.get("short_term_memory_size", 0)))
    table.add_row("Working Memory Active", str(status.get("working_memory_active", 0)))
    table.add_row("Knowledge Graph Nodes", str(status.get("semantic_knowledge_nodes", 0)))
    table.add_row("Active Modules", ", ".join(status.get("modules_active", [])))
    
    console.print(table)


def main():
    """Run all demos."""
    print_banner()
    
    console.print("[bold green]Starting Open AGI Brain Demo...[/bold green]\n")
    
    demos = [
        ("Basic Conversation", demo_basic_conversation),
        ("Memory System", demo_memory_system),
        ("Curiosity System", demo_curiosity_system),
        ("Self-Reflection", demo_self_reflection),
        ("System Status", demo_system_status),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            console.print(f"[red]Demo '{name}' error: {e}[/red]")
    
    console.print("\n[bold green]🧠 Demo complete! Open AGI Brain is running.[/bold green]")


if __name__ == "__main__":
    main()
