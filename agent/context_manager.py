import ast
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Set

from config import (
    MAX_CONTEXT_TOKENS,
    TOP_K_FILES,
    CONTEXT_WINDOW_LINES,
    MAX_WINDOWS_PER_FILE,
    MAX_CHUNKS_PER_FILE,
    ENABLE_EMBEDDINGS,
    ENABLE_PARALLEL,
    PARALLEL_THREADS,
    INDEX_FILE_PATH,
    SCORING_WEIGHTS,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    ENABLE_FTS_RETRIEVAL,
    ENABLE_CODE_GRAPH,
    CODE_GRAPH_SCAN_LIMIT,
)
from llm.client import LLMClient
from agent.code_graph import CodeGraphIndex
from mcp_server.retrieval_index import RetrievalIndex
from mcp_server.tools import read_file

STOPWORDS = {"the", "is", "at", "which", "on", "in", "a", "an", "and", "or", "for", "to", "with", "how", "what", "why", "where"}
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def estimate_tokens(text: str) -> int:
    words = text.split()
    punct_count = sum(1 for c in text if c in ".,(){}[]:;'\"`!=+-<>/*&|^%#@~")
    return max(1, len(words) + (punct_count // 2))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(x * y for x, y in zip(v1, v2))
    norm1 = sum(x * x for x in v1) ** 0.5
    norm2 = sum(x * x for x in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def extract_python_symbols(content: str) -> Set[str]:
    symbols: Set[str] = set()
    try:
        tree = ast.parse(content)
    except Exception:
        return symbols
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name.lower())
    return symbols


class ContextManager:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.index: Dict[str, Dict[str, Any]] = {}
        self.lexical_index = RetrievalIndex()
        self.code_graph = CodeGraphIndex()
        if ENABLE_EMBEDDINGS:
            self._load_index()

    def _load_index(self):
        if os.path.exists(INDEX_FILE_PATH):
            try:
                with open(INDEX_FILE_PATH, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
            except Exception:
                self.index = {}

    def _save_index(self):
        try:
            with open(INDEX_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.index, f)
        except Exception:
            pass

    def analyze_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["explain", "how", "what", "describe", "understand"]):
            return "explain"
        if any(w in q for w in ["bug", "error", "fail", "issue", "fix", "wrong"]):
            return "debug"
        return "search"

    def _lexical_scores(self, query_terms: Set[str]) -> Dict[str, float]:
        if not ENABLE_FTS_RETRIEVAL or not query_terms:
            return {}
        query_text = " OR ".join(sorted(query_terms))
        pairs = self.lexical_index.query(query_text, limit=max(20, TOP_K_FILES * RETRIEVAL_CANDIDATE_MULTIPLIER * 2))
        if not pairs:
            return {}
        # BM25 in sqlite FTS5: smaller score is better, often negative.
        min_score = min(score for _, score in pairs)
        max_score = max(score for _, score in pairs)
        denom = max(1e-9, max_score - min_score)
        normalized: Dict[str, float] = {}
        for path, score in pairs:
            raw = 1.0 - ((score - min_score) / denom)
            normalized[path] = max(0.0, raw)
        return normalized

    def rank_and_extract(self, file_infos: List[Dict[str, Any]], query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()
        intent = self.analyze_intent(query)
        q_lower = query.lower()
        query_terms = {t for t in TOKEN_RE.findall(q_lower) if t not in STOPWORDS}
        if not query_terms:
            query_terms = set(TOKEN_RE.findall(q_lower))

        query_embedding = None
        if ENABLE_EMBEDDINGS:
            query_embedding = self.llm.get_embedding(query)

        current_time = time.time()
        lexical_scores = self._lexical_scores(query_terms)
        graph_refresh_count = 0
        graph_boosts: Dict[str, float] = {}

        if ENABLE_CODE_GRAPH:
            py_infos = [fi for fi in file_infos if fi["path"].lower().endswith(".py")][:CODE_GRAPH_SCAN_LIMIT]
            payloads = []
            for info in py_infos:
                content = read_file(info["path"])
                if content is None:
                    continue
                payloads.append({"path": info["path"], "mtime": info["mtime"], "content": content})
            if payloads:
                graph_refresh_count = self.code_graph.upsert_python_files(payloads)
            graph_boosts = self.code_graph.query_file_boosts(sorted(query_terms), limit=max(50, TOP_K_FILES * 20))

        def cheap_rank(info: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
            f = info["path"]
            mtime = info["mtime"]
            age_days = (current_time - mtime) / (60 * 60 * 24)
            recency_score = max(0, SCORING_WEIGHTS["recency_bias_max"] - (age_days * 0.5))

            filename_val = SCORING_WEIGHTS["filename_match"]
            if intent == "explain":
                filename_val *= 1.5
            elif intent == "debug":
                recency_score *= 2.0

            score = recency_score
            file_name_lower = os.path.basename(f).lower()
            path_lower = f.lower()

            for term in query_terms:
                if term in file_name_lower:
                    score += filename_val
                elif term in path_lower:
                    score += filename_val * 0.5

            if f in lexical_scores:
                score += lexical_scores[f] * SCORING_WEIGHTS.get("lexical_bm25_weight", 20.0)
            if ENABLE_CODE_GRAPH and f in graph_boosts:
                score += graph_boosts[f]

            return info, score

        cheap_scores = [cheap_rank(info) for info in file_infos]
        cheap_scores.sort(key=lambda x: (-x[1], x[0]["path"]))
        candidate_count = min(len(file_infos), max(TOP_K_FILES, TOP_K_FILES * RETRIEVAL_CANDIDATE_MULTIPLIER))
        candidates = cheap_scores[:candidate_count]

        file_contents: Dict[str, str] = {}
        scored_files: List[Tuple[str, float]] = []
        index_updated = False
        unreadable_count = 0
        index_payloads: List[Dict[str, Any]] = []

        def process_file(candidate: Tuple[Dict[str, Any], float]) -> Dict[str, Any]:
            info, base_score = candidate
            f = info["path"]
            mtime = info["mtime"]
            content = read_file(f)
            if content is None:
                return {"file": f, "readable": False}

            content_lower = content.lower()
            content_len = max(1, len(content_lower))
            token_counts = Counter(TOKEN_RE.findall(content_lower))
            file_name_lower = os.path.basename(f).lower()
            score = base_score

            for term in query_terms:
                if term in file_name_lower:
                    score += SCORING_WEIGHTS["filename_match"] * 0.1
                score += (min(token_counts.get(term, 0), 15) * SCORING_WEIGHTS["keyword_match_mult"]) / content_len

            symbol_hits = 0
            if f.lower().endswith(".py"):
                symbols = extract_python_symbols(content)
                symbol_hits = sum(1 for term in query_terms if term in symbols)
                score += symbol_hits * SCORING_WEIGHTS.get("symbol_match", 80.0)

            embedding = None
            if ENABLE_EMBEDDINGS:
                cache_key = f
                if cache_key in self.index and self.index[cache_key].get("mtime") == mtime:
                    embedding = self.index[cache_key].get("embedding")
                else:
                    embedding = self.llm.get_embedding(content[:2000] + " " + file_name_lower)

            return {
                "file": f,
                "content": content,
                "score": score,
                "embedding": embedding,
                "mtime": mtime,
                "symbol_hits": symbol_hits,
                "readable": True,
            }

        if ENABLE_PARALLEL and candidates:
            with ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
                results = list(executor.map(process_file, candidates))
        else:
            results = [process_file(c) for c in candidates]

        for res in results:
            if not res.get("readable"):
                unreadable_count += 1
                continue
            f = res["file"]
            file_contents[f] = res["content"]
            score = res["score"]

            if ENABLE_EMBEDDINGS and query_embedding is not None and res["embedding"]:
                score += cosine_similarity(query_embedding, res["embedding"]) * SCORING_WEIGHTS["embedding_weight"]
                self.index[f] = {"mtime": res["mtime"], "embedding": res["embedding"]}
                index_updated = True

            scored_files.append((f, score))
            if ENABLE_FTS_RETRIEVAL:
                index_payloads.append({"path": f, "mtime": res["mtime"], "text_content": res["content"][:12000]})

        if index_updated:
            self._save_index()
        if ENABLE_FTS_RETRIEVAL and index_payloads:
            self.lexical_index.upsert_files(index_payloads)

        scored_files.sort(key=lambda x: (-x[1], x[0]))
        top_files = scored_files[:TOP_K_FILES]

        t_scoring = time.perf_counter() - t0
        t0_extract = time.perf_counter()

        extracted_chunks = []
        current_tokens = 0
        total_chars_used = 0
        selected_stats = []
        low_score_exclusions = 0
        top_score_sum = sum(s for _, s in top_files)

        score_total = max(1e-9, sum(max(s, 0.0) for _, s in top_files))
        for i, (f, score) in enumerate(top_files):
            if score < 5.0 and top_score_sum > 10.0:
                low_score_exclusions += 1
                continue

            # Adaptive token budget by relevance.
            weight = max(0.0, score) / score_total
            file_budget = max(300, int(MAX_CONTEXT_TOKENS * (0.15 + 0.75 * weight)))

            content = file_contents[f]
            lines = content.split("\n")
            content_lower_lines = [line.lower() for line in lines]

            relevant_indices = []
            for idx, lower_line in enumerate(content_lower_lines):
                if any(term in lower_line for term in query_terms):
                    relevant_indices.append(idx)

            # Symbol-aware fallback for Python files.
            if not relevant_indices and f.lower().endswith(".py"):
                try:
                    tree = ast.parse(content)
                    symbol_lines = []
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            symbol_lines.append(max(0, (node.lineno or 1) - 1))
                    relevant_indices = sorted(set(symbol_lines[:20]))
                except Exception:
                    pass

            chunks = []
            if not relevant_indices:
                chunk_lines = lines[: CONTEXT_WINDOW_LINES * 2]
                chunks.append("\n".join([f"{idx+1}: {line}" for idx, line in enumerate(chunk_lines)]))
            else:
                windows = []
                for idx in relevant_indices:
                    start = max(0, idx - CONTEXT_WINDOW_LINES)
                    end = min(len(lines), idx + CONTEXT_WINDOW_LINES + 1)
                    windows.append((start, end))

                windows.sort()
                merged = []
                for current in windows:
                    if not merged:
                        merged.append(current)
                    else:
                        prev = merged[-1]
                        if current[0] <= prev[1]:
                            merged[-1] = (prev[0], max(prev[1], current[1]))
                        else:
                            merged.append(current)
                merged = merged[:MAX_WINDOWS_PER_FILE]

                for start, end in merged:
                    win_lines = [f"{idx+start+1}: {line}" for idx, line in enumerate(lines[start:end])]
                    chunks.append(f"... Lines {start+1}-{end} ...\n" + "\n".join(win_lines))

            chunks = chunks[:MAX_CHUNKS_PER_FILE]

            priority_label = "[HIGH PRIORITY] " if i < 3 else ""
            combined_parts = []
            used_for_file = 0
            for chunk in chunks:
                chunk_tokens = estimate_tokens(chunk)
                if used_for_file + chunk_tokens > file_budget:
                    break
                combined_parts.append(chunk)
                used_for_file += chunk_tokens

            combined_file_chunk = (
                "========== FILE ==========\n"
                f"{priority_label}[FILE: {f} | SCORE: {score:.2f}]\n"
                + "\n...\n".join(combined_parts)
                + "\n==========================\n"
            )
            block_tokens = estimate_tokens(combined_file_chunk)
            if current_tokens + block_tokens > MAX_CONTEXT_TOKENS:
                break

            extracted_chunks.append({"file": f, "score": score, "content": combined_file_chunk, "tokens": block_tokens})
            current_tokens += block_tokens
            total_chars_used += len(combined_file_chunk)
            selected_stats.append({"file": f, "score": round(score, 2), "tokens": block_tokens, "budget": file_budget})

        t_extraction = time.perf_counter() - t0_extract

        total_low_score_exclusions = low_score_exclusions + sum(1 for _, s in scored_files[TOP_K_FILES:] if s < 5.0)
        trace_data = {
            "selected_files": selected_stats,
            "discarded_files_count_outside_topk": max(0, len(file_infos) - TOP_K_FILES),
            "low_score_exclusions": total_low_score_exclusions,
            "retrieval_candidates_considered": len(candidates),
            "unreadable_files_skipped": unreadable_count,
            "lexical_hits": len(lexical_scores),
            "code_graph_boosted_files": len(graph_boosts),
            "code_graph_refreshed_files": graph_refresh_count,
            "estimated_tokens_used": current_tokens,
            "total_chars_used": total_chars_used,
            "query_intent": intent,
            "execution_times": {"scoring": round(t_scoring, 3), "extraction": round(t_extraction, 3)},
        }
        return extracted_chunks, trace_data
