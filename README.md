# Athena MCP Agent

**Athena** is a high-performance coding agent CLI designed for **grounded code understanding**, **autonomous patch execution**, and **safe validation loops**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![No Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)](#requirements)

---

## Overview

Athena goes beyond simple code Q&A. It is a full agentic operator that understands your repository's structure, proposes complex changes, and validates them iteratively until the goal is achieved.

*   **Grounded Analysis**: Deep lexical and graph-based indexing for accurate retrieval.
*   **Autonomous Patching**: Parallel candidate generation and iterative validation.
*   **Safety First**: Workspace confinement, risk gating, and automatic rollbacks.
*   **Zero Dependencies**: Built entirely on Python standard libraries for maximum portability.

---

## Key Features

*   **Retrieval-Augmented Analysis**: `debug` and `explain` commands powered by SQLite FTS5 and AST code graphs.
*   **Iterative Autofix**: Autonomous plan-edit-validate loops that hunt for bugs and implement features.
*   **Diff-Focused Review**: A dedicated `review` agent to analyze local or staged git diffs.
*   **Git Integration**: Automatic branching, commit drafting, and safe rollback strategies.
*   **Evaluation Flywheel**: Run persistence evaluation tasks (`evaluate`) to ensure no regressions.
*   **Operator Mode**: A long-running interactive `session` for continuous pair-programming.

---

## Installation & Setup

### Prerequisites
*   Python 3.10+
*   SQLite with FTS5 support (standard in most Python builds)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Lightningxyz/athena-mcp-agent.git
cd athena-mcp-agent

# No pip install needed! Athena uses zero third-party libraries.

# Setup your environment
# Create a .env file and add your API keys (see Configuration below)
```

### Configuration
Edit your `.env` file to configure LLM providers:
```bash
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
# Optional: Use local LLMs via Ollama
LLM_API_BASE=http://localhost:11434/v1
```

---

## Usage Guide

### 1. `debug` & `explain`
Ask questions or get deep dives into specific paths.
```bash
python3 main.py debug "How is the session memory persisted?" --path .
python3 main.py explain agent/core.py
```

### 2. `autofix` (The Agentic Loop)
Run iterative loops to solve problems.
```bash
python3 main.py autofix "Fix the flaky unit tests in agent/core.py" \
  --path . \
  --apply \
  --validate-cmd "python3 -m unittest tests/test_core.py"
```

### 3. `review`
Analyze your changes before committing.
```bash
python3 main.py review --staged
```

### 4. `session`
Start an interactive pair-programming session.
```bash
python3 main.py session --path . --apply --git
```

---

## Safety Model

Athena is designed for safe local execution:
*   **Path Confinement**: Never edits files outside the target repository.
*   **Risk Gating**: High-risk operations require explicit `--approve-high-risk` flags.
*   **Command Filtering**: Validation commands are checked against allowlists and blocklists.
*   **Safe Rollbacks**: If a patch fails validation, Athena automatically restores the previous state.

---

## Project Structure

*   `agent/`: Core agent logic (Autofix, Planning, Review)
*   `llm/`: LLM client with routing and circuit breakers
*   `mcp_server/`: Indexing and retrieval tools
*   `utils/`: Safety policies and Git primitives
*   `eval/`: Regression testing and benchmarking
*   `tests/`: Project test suite

---


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
