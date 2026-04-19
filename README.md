# Athena MCP Agent

Athena is a local-first coding agent CLI focused on **grounded code understanding**, **autonomous patch execution**, and **safe validation loops**.

It is designed to move beyond simple code Q&A by supporting:

- Retrieval-augmented analysis (`debug`, `explain`)
- Iterative autonomous patching (`autofix`)
- Diff-focused review agent (`review`)
- Git-aware automation (branching, commit drafting, commit automation)
- Evaluation flywheel (`evaluate`) with regression gates
- Interactive long-running operator mode (`session`)

---

## What It Does Today

Athena runs a multi-stage engineering workflow:

1. Repository indexing and retrieval
- File scanning with extension/size filters
- Lexical index (SQLite FTS5/BM25)
- Code graph index (symbols/imports/calls, Python AST)
- Optional embedding signals

2. Model routing
- Health-aware routing across providers/models
- Retry, backoff, circuit-breakers, and local fallback
- Usage/cost estimation for every LLM call

3. Structured reasoning
- Deterministic output schema enforcement (`Answer` + `Justification`)
- Self-evaluation and repair passes
- Confidence signaling (`Low`, `Medium`, `High`)

4. Agentic patch loop (`autofix`)
- LLM generates multiple strict JSON patch plan candidates in parallel
- AST-aware patch engine (`replace_symbol`) + deterministic snippet edits
- Extended patch ops: `create_file`, `delete_file`, `append_snippet`
- Dry-run preview with unified diffs
- Safe apply with syntax guards and rollback
- Per-iteration file checkpoints with restore on validation failure
- Validation command execution with safety allowlist/blocklist
- Automatic validation command detection (`python`, `npm`, `go`, `cargo`) when none are provided
- Persistent session memory across runs
- Run artifact JSON outputs for postmortem/replay
- Candidate scoring (heuristic + LLM evaluator) chooses best plan before apply

5. Git automation
- Optional branch creation
- Diff stats and commit automation
- PR draft text generation in run report

---

## CLI Commands

### `debug`
Ask a question about a codebase.

```bash
python3 main.py debug "Where is auth token refresh handled?" --path .
```

### `explain`
Explain a codebase path.

```bash
python3 main.py explain .
```

### `autofix`
Run iterative plan-edit-validate loops.

Dry run:

```bash
python3 main.py autofix "Fix failing tests in context manager" --path . --max-iterations 3
```

Apply edits with safety and validation:

```bash
python3 main.py autofix "Fix failing tests in context manager" \
  --path . \
  --apply \
  --max-iterations 4 \
  --plan-candidates 4 \
  --validate-cmd "python3 -m unittest discover -s tests -p test*.py -v"
```

### `review`
Run strict code-review analysis over your local git diff:

```bash
python3 main.py review --path .
```

Review staged changes only:

```bash
python3 main.py review --path . --staged
```

Review against a base ref with focus hint:

```bash
python3 main.py review --path . --base-ref origin/main --focus "security and regressions"
```

Enable git workflow:

```bash
python3 main.py autofix "Fix flaky tests and clean imports" \
  --path . \
  --apply \
  --git \
  --create-branch \
  --auto-commit
```

### `evaluate`
Run persistent evaluation tasks and enforce regression gates.

```bash
python3 main.py evaluate --tasks eval/tasks.json
```

### `session`
Run Athena as an interactive operator loop:

```bash
python3 main.py session --path . --apply --git
```

Inside session:
- `debug: <query>`
- `explain`
- `autofix: <goal>`
- `exit`

---

## Safety Model

Autofix safety includes:

- Workspace path confinement (no edits outside target root)
- High-risk gating (`--approve-high-risk` required when configured)
- Validation command allowlist and blocked-term policy
- Rollback on failed patch application
- Non-destructive git rollback strategy (`revert`, not hard reset)

---

## Project Structure

```text
agent/
  core.py               # Q&A orchestration and structured response loop
  autofix.py            # Plan-edit-validate agent loop
  patch_engine.py       # AST-aware + deterministic patch application
  context_manager.py    # Retrieval ranking and adaptive context extraction
  code_graph.py         # Incremental code graph (symbols/imports/calls)
  session_memory.py     # Persistent session memory
  output_schema.py      # Output schema parser/repair helpers

llm/
  client.py             # Routing, retries, circuit breakers, usage accounting

mcp_server/
  tools.py              # File scan/read tools
  retrieval_index.py    # SQLite FTS lexical index

utils/
  safety.py             # Command/path/risk safety policy
  git_ops.py            # Git automation primitives

eval/
  runner.py             # Evaluation flywheel + regression gates
  tasks.json            # Eval task set

tests/
  test_core.py          # Unit + integration coverage
```

---

## Requirements

- Python 3.10+
- SQLite with FTS5 support (standard in most modern Python builds)

No third-party dependencies are required at this time.

---

## Setup

```bash
git clone <your-repo-url>
cd mcp
```

Optional `.env` example:

```bash
GROQ_API_KEY=
OPENROUTER_API_KEY=
LLM_API_BASE=http://localhost:11434/v1

ENABLE_EMBEDDINGS=false
ENABLE_FTS_RETRIEVAL=true
ENABLE_CODE_GRAPH=true

AUTOFIX_REQUIRE_APPROVAL_FOR_HIGH_RISK=true
AUTOFIX_ALLOWED_COMMAND_PREFIXES=python3 -m unittest,pytest,npm test
```

---

## Testing

```bash
python3 -m unittest discover -s tests -p 'test*.py' -v
python3 -m py_compile $(rg --files -g '*.py')
```

---

## Notes

Athena is now much closer to a full coding agent architecture (plan-edit-validate, safety policy, memory, graph retrieval, and git integration), but it is still evolving and should be treated as a high-capability tool with human supervision for production-critical changes.

All CLI runs append structured trace events to JSONL (`TRACE_JSONL_FILE`) and autofix runs write machine-readable artifacts under `RUN_ARTIFACTS_DIR`.
