# 🧠 Open AGI Brain — Artificial Cognitive Architecture

> **An Operating System for Intelligence**
> A modular AI cognitive architecture that mimics how the human brain organizes intelligence.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Modular_AGI-purple.svg)]()
[![Status](https://img.shields.io/badge/Status-Research_Project-orange.svg)]()

---

## 🌐 Overview

Most AI systems today (ChatGPT, Claude, Gemini) are **single-model systems**.

**Open AGI Brain** is different — it is a **modular intelligence architecture** where multiple specialized AI engines work together, just like regions of the human brain coordinate to produce thought, memory, and action.

This is how future AGI systems may be built.

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OPEN AGI BRAIN FRAMEWORK                        │
│                    (Cognitive Orchestrator)                         │
├────────────┬──────────────┬─────────────┬───────────────────────────┤
│ PERCEPTION │  REASONING   │   MEMORY    │     CURIOSITY SYSTEM      │
│  ENGINE    │   ENGINE     │   SYSTEM    │                           │
│            │              │             │  • Novelty Detection      │
│ • Text NLP │ • Symbolic   │ • Short-term│  • Intrinsic Reward       │
│ • Vision   │   Reasoning  │ • Working   │  • Exploration RL         │
│ • Audio    │ • Chain-of-  │ • Long-term │                           │
│ • Sensors  │   Thought    │ • Episodic  │                           │
│            │ • Causal Inf │ • Semantic  │                           │
├────────────┴──────────────┼─────────────┴───────────────────────────┤
│      DECISION ENGINE      │        SELF-REFLECTION MODULE           │
│                           │                                         │
│  • Goal-based Planning    │  1. Generate Answer                     │
│  • Risk Estimation        │  2. Critic Evaluates                    │
│  • Reinforcement Learning │  3. Improved Answer Produced            │
└───────────────────────────┴─────────────────────────────────────────┘
```

---

## 🧩 Module Details

### 1️⃣ Perception Engine (`modules/perception/`)

Processes input from multiple modalities:

- **Text** — NLP models (transformers, BERT, LLMs)
- **Images** — Vision models (CLIP, ViT, YOLO)
- **Audio** — Speech recognition (OpenAI Whisper)
- **Sensor data** — Multimodal fusion pipelines

### 2️⃣ Reasoning Engine (`modules/reasoning/`)

Handles logic and complex problem-solving:

- Symbolic reasoning (rule-based logic)
- Chain-of-Thought (CoT) reasoning
- Planning and goal decomposition
- Causal inference

### 3️⃣ Memory System (`modules/memory/`)

Human-like hierarchical memory architecture:

| Memory Type | Description | Implementation |
|-------------|-------------|----------------|
| Short-term | Immediate context buffer | In-memory queue (deque) |
| Working | Active task workspace | Redis / Python dict |
| Long-term | Persistent knowledge store | Vector DB (ChromaDB) |
| Episodic | Past experience recall | ChromaDB + timestamps |
| Semantic | World knowledge & facts | Knowledge Graphs (NetworkX) |

### 4️⃣ Curiosity System (`modules/curiosity/`)

AI-driven exploration and discovery engine:

- Intrinsic reward signals for novel inputs
- Novelty detection via cosine distance on embeddings
- Reinforcement learning-based exploration policy

### 5️⃣ Decision Engine (`modules/decision/`)

Intelligent action selection based on goals:

- Goal-based planning (BFS, A*, MCTS)
- Risk estimation and confidence scoring
- Reinforcement learning policy networks

### 6️⃣ Self-Reflection Module (`modules/self_reflection/`)

Meta-cognitive evaluation loop:

```
Input Query
    ↓
Generate Initial Answer  (LLM pass 1)
    ↓
Critic Evaluates Answer  (LLM pass 2 - evaluate quality, gaps, errors)
    ↓
Generate Improved Answer (LLM pass 3 - incorporate feedback)
    ↓
Final Output
```

---

## 📁 Project Structure

```
open-agi-brain/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml              # Global configuration
├── core/
│   └── orchestrator.py            # Central cognitive orchestrator
├── modules/
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── text_processor.py      # NLP/text perception
│   │   ├── vision_processor.py    # Image perception (CLIP/ViT)
│   │   └── audio_processor.py     # Speech/audio (Whisper)
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── symbolic.py            # Symbolic/rule-based reasoning
│   │   ├── chain_of_thought.py    # CoT prompting
│   │   └── causal.py              # Causal inference
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py          # Short-term memory buffer
│   │   ├── working_memory.py      # Working memory workspace
│   │   ├── long_term.py           # Long-term vector DB memory
│   │   ├── episodic.py            # Episodic memory (past events)
│   │   └── semantic.py            # Semantic knowledge graph
│   ├── curiosity/
│   │   ├── __init__.py
│   │   └── curiosity_engine.py    # Novelty & exploration engine
│   ├── decision/
│   │   ├── __init__.py
│   │   └── decision_engine.py     # Goal-based decision making
│   └── self_reflection/
│       ├── __init__.py
│       └── reflection_module.py   # Meta-cognitive reflection
├── utils/
│   ├── embeddings.py              # Embedding utilities
│   └── logger.py                  # Structured logging
├── examples/
│   └── demo.py                    # End-to-end system demo
└── tests/
    └── test_modules.py            # Unit tests for all modules
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/PranayMahendrakar/open-agi-brain.git
cd open-agi-brain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python examples/demo.py
```

---

## 🔧 Configuration

Edit `config/settings.yaml` to configure:

```yaml
llm:
  provider: openai          # openai, anthropic, ollama
  model: gpt-4
  temperature: 0.7

memory:
  short_term_capacity: 10
  vector_db_path: ./data/chroma_db

curiosity:
  novelty_threshold: 0.3
  exploration_rate: 0.1
```

---

## 🔬 Research Areas Covered

- **Cognitive AI** — Modeling intelligence as modular, brain-inspired systems
- **Reinforcement Learning** — Curiosity-driven and goal-directed agents
- **Knowledge Graphs** — Semantic memory and relational reasoning
- **LLM Reasoning** — Chain-of-thought, self-consistency, reflection loops
- **Memory Systems** — Hierarchical memory structures analogous to human cognition

---

## 🛣️ Roadmap

- [x] Core architecture design
- [x] Perception Engine (text, vision, audio)
- [x] Reasoning Engine (CoT, symbolic, causal)
- [x] Memory System (5 memory types)
- [x] Curiosity System (novelty detection)
- [x] Decision Engine (planning + RL)
- [x] Self-Reflection Module (critic loop)
- [x] Orchestrator (module coordination)
- [ ] GUI Dashboard (real-time brain activity visualization)
- [ ] Multi-agent collaboration support
- [ ] Continual learning integration
- [ ] Benchmark evaluations (ARC, BIG-Bench, HellaSwag)
- [ ] Docker containerization
- [ ] REST API server

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-module`
3. Commit your changes: `git commit -m 'Add new module'`
4. Push to the branch: `git push origin feature/new-module`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Pranay M Mahendrakar**
AI Specialist & Innovator | SONYTECH
🌐 https://sonytech.in/pranay/
🔬 ORCID: https://orcid.org/0009-0003-7224-029X

---

> *"Intelligence is not a single thing — it is a symphony of many specialized systems working in harmony."*
