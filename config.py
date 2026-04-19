import os
from dataclasses import dataclass
from typing import Dict, Set


def _load_env_file() -> None:
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw}")


def _env_int(name: str, default: int, min_value: int = None, max_value: int = None) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc
    if min_value is not None and val < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {val}")
    if max_value is not None and val > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {val}")
    return val


def _env_float(name: str, default: float, min_value: float = None, max_value: float = None) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        val = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw}") from exc
    if min_value is not None and val < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {val}")
    if max_value is not None and val > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {val}")
    return val


def _env_csv(name: str, default_csv: str) -> str:
    raw = os.getenv(name, default_csv).strip()
    return raw


_load_env_file()


@dataclass(frozen=True)
class Settings:
    max_context_tokens: int
    chars_per_token: int
    top_k_files: int
    context_window_lines: int
    max_windows_per_file: int
    max_chunks_per_file: int
    max_file_bytes: int
    allowed_extensions: Set[str]
    ignore_dirs: Set[str]
    max_scan_files: int
    enable_embeddings: bool
    enable_parallel: bool
    parallel_threads: int
    retrieval_candidate_multiplier: int
    index_file_path: str
    lexical_index_db_path: str
    enable_fts_retrieval: bool
    scoring_weights: Dict[str, float]
    groq_api_key: str
    openrouter_api_key: str
    llm_api_base: str
    model_cache_ttl_seconds: int
    llm_max_retries: int
    llm_base_retry_delay_seconds: float
    llm_groq_timeout_seconds: int
    llm_openrouter_timeout_seconds: int
    llm_local_timeout_seconds: int
    llm_circuit_failure_threshold: int
    llm_circuit_reset_seconds: int
    default_input_token_cost_per_1k: float
    default_output_token_cost_per_1k: float
    code_graph_db_path: str
    enable_code_graph: bool
    code_graph_scan_limit: int
    session_memory_file: str
    session_memory_max_entries: int
    autofix_allowed_command_prefixes_csv: str
    autofix_blocked_command_terms_csv: str
    autofix_require_approval_for_high_risk: bool
    autofix_max_iterations: int
    git_auto_branch_prefix: str
    git_auto_commit: bool
    eval_history_file: str
    router_latency_weight: float
    router_failure_penalty: float
    run_artifacts_dir: str
    trace_jsonl_file: str
    checkpoints_dir: str
    autofix_plan_candidates: int
    patch_parallel_workers: int
    autofix_enable_review_gate: bool
    autofix_review_blocking_risks_csv: str


def _build_settings() -> Settings:
    return Settings(
        max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 8000, min_value=256),
        chars_per_token=_env_int("CHARS_PER_TOKEN", 4, min_value=1),
        top_k_files=_env_int("TOP_K_FILES", 5, min_value=1),
        context_window_lines=_env_int("CONTEXT_WINDOW_LINES", 10, min_value=1),
        max_windows_per_file=_env_int("MAX_WINDOWS_PER_FILE", 5, min_value=1),
        max_chunks_per_file=_env_int("MAX_CHUNKS_PER_FILE", 5, min_value=1),
        max_file_bytes=_env_int("MAX_FILE_BYTES", 1024 * 1024, min_value=1024),
        allowed_extensions={
            ".py", ".js", ".ts", ".json", ".md", ".txt", ".go", ".java", ".c", ".cpp", ".h", ".cs"
        },
        ignore_dirs={
            ".git", "__pycache__", "node_modules", "venv", ".venv", "env", "build", "dist", "target", "out"
        },
        max_scan_files=_env_int("MAX_SCAN_FILES", 5000, min_value=10),
        enable_embeddings=_env_bool("ENABLE_EMBEDDINGS", False),
        enable_parallel=_env_bool("ENABLE_PARALLEL", True),
        parallel_threads=_env_int("PARALLEL_THREADS", 8, min_value=1, max_value=64),
        retrieval_candidate_multiplier=_env_int("RETRIEVAL_CANDIDATE_MULTIPLIER", 6, min_value=1, max_value=50),
        index_file_path=os.getenv("INDEX_FILE_PATH", ".mcp_index.json"),
        lexical_index_db_path=os.getenv("LEXICAL_INDEX_DB_PATH", ".mcp_lexical_index.sqlite3"),
        enable_fts_retrieval=_env_bool("ENABLE_FTS_RETRIEVAL", True),
        scoring_weights={
            "filename_match": 50.0,
            "keyword_match_mult": 200.0,
            "recency_bias_max": 20.0,
            "embedding_weight": 100.0,
            "symbol_match": 80.0,
            "lexical_bm25_weight": 20.0,
        },
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        llm_api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        model_cache_ttl_seconds=_env_int("MODEL_CACHE_TTL_SECONDS", 300, min_value=1),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 3, min_value=1, max_value=10),
        llm_base_retry_delay_seconds=_env_float("LLM_BASE_RETRY_DELAY_SECONDS", 0.5, min_value=0.01, max_value=10.0),
        llm_groq_timeout_seconds=_env_int("LLM_GROQ_TIMEOUT_SECONDS", 35, min_value=1),
        llm_openrouter_timeout_seconds=_env_int("LLM_OPENROUTER_TIMEOUT_SECONDS", 45, min_value=1),
        llm_local_timeout_seconds=_env_int("LLM_LOCAL_TIMEOUT_SECONDS", 45, min_value=1),
        llm_circuit_failure_threshold=_env_int("LLM_CIRCUIT_FAILURE_THRESHOLD", 4, min_value=1, max_value=20),
        llm_circuit_reset_seconds=_env_int("LLM_CIRCUIT_RESET_SECONDS", 60, min_value=1),
        default_input_token_cost_per_1k=_env_float("DEFAULT_INPUT_TOKEN_COST_PER_1K", 0.001, min_value=0.0),
        default_output_token_cost_per_1k=_env_float("DEFAULT_OUTPUT_TOKEN_COST_PER_1K", 0.002, min_value=0.0),
        code_graph_db_path=os.getenv("CODE_GRAPH_DB_PATH", ".athena_code_graph.sqlite3"),
        enable_code_graph=_env_bool("ENABLE_CODE_GRAPH", True),
        code_graph_scan_limit=_env_int("CODE_GRAPH_SCAN_LIMIT", 3000, min_value=10),
        session_memory_file=os.getenv("SESSION_MEMORY_FILE", ".athena_session_memory.json"),
        session_memory_max_entries=_env_int("SESSION_MEMORY_MAX_ENTRIES", 100, min_value=5, max_value=5000),
        autofix_allowed_command_prefixes_csv=_env_csv(
            "AUTOFIX_ALLOWED_COMMAND_PREFIXES",
            "python3 -m unittest,pytest,npm test,npm run test,go test,cargo test",
        ),
        autofix_blocked_command_terms_csv=_env_csv(
            "AUTOFIX_BLOCKED_COMMAND_TERMS",
            "rm -rf,sudo,mkfs,dd if=,:(){,shutdown,reboot,halt,poweroff,git reset --hard",
        ),
        autofix_require_approval_for_high_risk=_env_bool("AUTOFIX_REQUIRE_APPROVAL_FOR_HIGH_RISK", True),
        autofix_max_iterations=_env_int("AUTOFIX_MAX_ITERATIONS", 4, min_value=1, max_value=20),
        git_auto_branch_prefix=os.getenv("GIT_AUTO_BRANCH_PREFIX", "athena/autofix"),
        git_auto_commit=_env_bool("GIT_AUTO_COMMIT", False),
        eval_history_file=os.getenv("EVAL_HISTORY_FILE", "eval/history.json"),
        router_latency_weight=_env_float("ROUTER_LATENCY_WEIGHT", 0.6, min_value=0.0, max_value=5.0),
        router_failure_penalty=_env_float("ROUTER_FAILURE_PENALTY", 1.8, min_value=0.0, max_value=10.0),
        run_artifacts_dir=os.getenv("RUN_ARTIFACTS_DIR", ".athena_runs"),
        trace_jsonl_file=os.getenv("TRACE_JSONL_FILE", ".athena_runs/trace.jsonl"),
        checkpoints_dir=os.getenv("CHECKPOINTS_DIR", ".athena_checkpoints"),
        autofix_plan_candidates=_env_int("AUTOFIX_PLAN_CANDIDATES", 3, min_value=1, max_value=8),
        patch_parallel_workers=_env_int("PATCH_PARALLEL_WORKERS", 4, min_value=1, max_value=32),
        autofix_enable_review_gate=_env_bool("AUTOFIX_ENABLE_REVIEW_GATE", True),
        autofix_review_blocking_risks_csv=_env_csv("AUTOFIX_REVIEW_BLOCKING_RISKS", "high,critical"),
    )


SETTINGS = _build_settings()

# Backward-compatible constants
MAX_CONTEXT_TOKENS = SETTINGS.max_context_tokens
CHARS_PER_TOKEN = SETTINGS.chars_per_token
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN
TOP_K_FILES = SETTINGS.top_k_files
CONTEXT_WINDOW_LINES = SETTINGS.context_window_lines
MAX_WINDOWS_PER_FILE = SETTINGS.max_windows_per_file
MAX_CHUNKS_PER_FILE = SETTINGS.max_chunks_per_file
MAX_FILE_BYTES = SETTINGS.max_file_bytes
ALLOWED_EXTENSIONS = SETTINGS.allowed_extensions
IGNORE_DIRS = SETTINGS.ignore_dirs
MAX_SCAN_FILES = SETTINGS.max_scan_files
ENABLE_EMBEDDINGS = SETTINGS.enable_embeddings
ENABLE_PARALLEL = SETTINGS.enable_parallel
PARALLEL_THREADS = SETTINGS.parallel_threads
RETRIEVAL_CANDIDATE_MULTIPLIER = SETTINGS.retrieval_candidate_multiplier
INDEX_FILE_PATH = SETTINGS.index_file_path
LEXICAL_INDEX_DB_PATH = SETTINGS.lexical_index_db_path
ENABLE_FTS_RETRIEVAL = SETTINGS.enable_fts_retrieval
SCORING_WEIGHTS = SETTINGS.scoring_weights
GROQ_API_KEY = SETTINGS.groq_api_key
OPENROUTER_API_KEY = SETTINGS.openrouter_api_key
LLM_API_BASE = SETTINGS.llm_api_base
MODEL_CACHE_TTL_SECONDS = SETTINGS.model_cache_ttl_seconds
LLM_MAX_RETRIES = SETTINGS.llm_max_retries
LLM_BASE_RETRY_DELAY_SECONDS = SETTINGS.llm_base_retry_delay_seconds
LLM_GROQ_TIMEOUT_SECONDS = SETTINGS.llm_groq_timeout_seconds
LLM_OPENROUTER_TIMEOUT_SECONDS = SETTINGS.llm_openrouter_timeout_seconds
LLM_LOCAL_TIMEOUT_SECONDS = SETTINGS.llm_local_timeout_seconds
LLM_CIRCUIT_FAILURE_THRESHOLD = SETTINGS.llm_circuit_failure_threshold
LLM_CIRCUIT_RESET_SECONDS = SETTINGS.llm_circuit_reset_seconds
DEFAULT_INPUT_TOKEN_COST_PER_1K = SETTINGS.default_input_token_cost_per_1k
DEFAULT_OUTPUT_TOKEN_COST_PER_1K = SETTINGS.default_output_token_cost_per_1k
CODE_GRAPH_DB_PATH = SETTINGS.code_graph_db_path
ENABLE_CODE_GRAPH = SETTINGS.enable_code_graph
CODE_GRAPH_SCAN_LIMIT = SETTINGS.code_graph_scan_limit
SESSION_MEMORY_FILE = SETTINGS.session_memory_file
SESSION_MEMORY_MAX_ENTRIES = SETTINGS.session_memory_max_entries
AUTOFIX_ALLOWED_COMMAND_PREFIXES_CSV = SETTINGS.autofix_allowed_command_prefixes_csv
AUTOFIX_BLOCKED_COMMAND_TERMS_CSV = SETTINGS.autofix_blocked_command_terms_csv
AUTOFIX_REQUIRE_APPROVAL_FOR_HIGH_RISK = SETTINGS.autofix_require_approval_for_high_risk
AUTOFIX_MAX_ITERATIONS = SETTINGS.autofix_max_iterations
GIT_AUTO_BRANCH_PREFIX = SETTINGS.git_auto_branch_prefix
GIT_AUTO_COMMIT = SETTINGS.git_auto_commit
EVAL_HISTORY_FILE = SETTINGS.eval_history_file
ROUTER_LATENCY_WEIGHT = SETTINGS.router_latency_weight
ROUTER_FAILURE_PENALTY = SETTINGS.router_failure_penalty
RUN_ARTIFACTS_DIR = SETTINGS.run_artifacts_dir
TRACE_JSONL_FILE = SETTINGS.trace_jsonl_file
CHECKPOINTS_DIR = SETTINGS.checkpoints_dir
AUTOFIX_PLAN_CANDIDATES = SETTINGS.autofix_plan_candidates
PATCH_PARALLEL_WORKERS = SETTINGS.patch_parallel_workers
AUTOFIX_ENABLE_REVIEW_GATE = SETTINGS.autofix_enable_review_gate
AUTOFIX_REVIEW_BLOCKING_RISKS_CSV = SETTINGS.autofix_review_blocking_risks_csv
