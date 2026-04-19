import asyncio
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

from config import (
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    LLM_API_BASE,
    MODEL_CACHE_TTL_SECONDS,
    LLM_MAX_RETRIES,
    LLM_BASE_RETRY_DELAY_SECONDS,
    LLM_GROQ_TIMEOUT_SECONDS,
    LLM_OPENROUTER_TIMEOUT_SECONDS,
    LLM_LOCAL_TIMEOUT_SECONDS,
    LLM_CIRCUIT_FAILURE_THRESHOLD,
    LLM_CIRCUIT_RESET_SECONDS,
    DEFAULT_INPUT_TOKEN_COST_PER_1K,
    DEFAULT_OUTPUT_TOKEN_COST_PER_1K,
    ROUTER_LATENCY_WEIGHT,
    ROUTER_FAILURE_PENALTY,
)


@dataclass
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


class LLMClient:
    def __init__(self):
        self._model_cache: Dict[str, Dict[str, Any]] = {
            "groq": {"models": None, "fetched_at": 0.0},
            "openrouter": {"models": None, "fetched_at": 0.0},
        }
        self._groq_models = None
        self._openrouter_models = None
        self._last_error: Optional[Dict[str, Any]] = None
        self._last_usage: Optional[LLMUsage] = None
        self._circuit_state: Dict[str, Dict[str, float]] = {
            "groq": {"failures": 0.0, "opened_at": 0.0},
            "openrouter": {"failures": 0.0, "opened_at": 0.0},
            "local": {"failures": 0.0, "opened_at": 0.0},
        }
        self._model_perf: Dict[str, Dict[str, float]] = {}

    def get_available_models(self, provider: str) -> List[str]:
        if provider == "groq":
            if self._groq_models is not None:
                return self._groq_models
            if not GROQ_API_KEY:
                return []
            cache = self._model_cache["groq"]
            if cache["models"] is not None and (time.time() - cache["fetched_at"]) < MODEL_CACHE_TTL_SECONDS:
                return cache["models"]
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    models = [m["id"] for m in data.get("data", [])]
                    cache["models"] = models
                    cache["fetched_at"] = time.time()
                    return models
            except Exception as exc:
                self._last_error = {
                    "type": "model_discovery_error",
                    "provider": provider,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return cache["models"] or []

        if provider == "openrouter":
            if self._openrouter_models is not None:
                return self._openrouter_models
            if not OPENROUTER_API_KEY:
                return []
            cache = self._model_cache["openrouter"]
            if cache["models"] is not None and (time.time() - cache["fetched_at"]) < MODEL_CACHE_TTL_SECONDS:
                return cache["models"]
            req = urllib.request.Request("https://openrouter.ai/api/v1/models")
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    models = [m["id"] for m in data.get("data", [])]
                    cache["models"] = models
                    cache["fetched_at"] = time.time()
                    return models
            except Exception as exc:
                self._last_error = {
                    "type": "model_discovery_error",
                    "provider": provider,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return cache["models"] or []

        return []

    def _preferred_candidates(self, task_type: str) -> List[Tuple[str, str]]:
        if task_type in ["fast", "compression", "scan"]:
            return [
                ("groq", "llama-3.3-70b-versatile"),
                ("groq", "mixtral-8x7b-32768"),
                ("groq", "llama3-70b-8192"),
                ("groq", "llama3-8b-8192"),
                ("local", "llama3"),
            ]
        if task_type in ["plan", "refactor", "patch", "strong", "answer", "eval"]:
            return [
                ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
                ("openrouter", "deepseek/deepseek-chat"),
                ("openrouter", "qwen/qwen-2.5-72b-instruct"),
                ("openrouter", "anthropic/claude-3.5-sonnet"),
                ("groq", "llama-3.3-70b-versatile"),
                ("local", "llama3"),
            ]
        return [("openrouter", ""), ("groq", ""), ("local", "llama3")]

    def route_model(self, task_type: str, quality: str = "balanced") -> Tuple[str, str]:
        candidates = self._preferred_candidates(task_type)
        available = {
            "groq": set(self.get_available_models("groq")),
            "openrouter": set(self.get_available_models("openrouter")),
            "local": {"llama3"},
        }

        best = ("local", "llama3")
        best_score = float("inf")
        for rank, (provider, model) in enumerate(candidates):
            if self._is_circuit_open(provider):
                continue
            if model and model not in available.get(provider, set()):
                continue
            chosen_model = model or next(iter(available.get(provider, {"llama3"})), "llama3")
            key = f"{provider}/{chosen_model}"
            perf = self._model_perf.get(key, {})
            avg_latency = perf.get("avg_latency", 1.5)
            failures = perf.get("failures", 0.0)
            calls = max(1.0, perf.get("calls", 1.0))
            failure_rate = failures / calls

            quality_bias = 0.0
            if quality == "quality":
                quality_bias = -0.2 * (1.0 / (1 + rank))
            elif quality == "speed":
                quality_bias = 0.2 * rank

            score = (
                rank
                + (avg_latency * ROUTER_LATENCY_WEIGHT)
                + (failure_rate * ROUTER_FAILURE_PENALTY)
                + quality_bias
            )
            if score < best_score:
                best_score = score
                best = (provider, chosen_model)

        return best

    def select_best_model(self, task_type: str) -> Tuple[str, str]:
        return self.route_model(task_type, quality="balanced")

    async def query_model_async(self, provider: str, model: str, prompt: str, system: str = "") -> str:
        errors: List[str] = []
        start_provider = provider
        start_model = model

        if self._is_circuit_open(provider):
            errors.append(f"{provider}/{model}: circuit open")
            provider, model = ("local", "llama3")

        for attempt in range(LLM_MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                res = await self._execute_query_async(provider, model, prompt, system)
                if not res or not res.strip():
                    errors.append(f"{provider}/{model} attempt {attempt + 1}: empty response")
                    delay = self._retry_delay(attempt)
                    await asyncio.sleep(delay)
                    self._record_model_metric(provider, model, success=False, latency=(time.perf_counter() - t0))
                    continue
                self._mark_success(provider)
                self._record_model_metric(provider, model, success=True, latency=(time.perf_counter() - t0))
                if errors:
                    self._last_error = {
                        "type": "transient_retry_recovered",
                        "provider": provider,
                        "model": model,
                        "errors": errors[-3:],
                    }
                else:
                    self._last_error = None
                self._last_usage = self._estimate_usage(prompt, res)
                return res
            except urllib.error.HTTPError as exc:
                self._mark_failure(provider)
                self._record_model_metric(provider, model, success=False, latency=0.0)
                errors.append(f"{provider}/{model} attempt {attempt + 1}: HTTPError: {exc.code}")
                if exc.code == 429:
                    await asyncio.sleep(self._retry_delay(attempt, rate_limited=True))
                    continue
                await asyncio.sleep(self._retry_delay(attempt))
            except Exception as exc:
                self._mark_failure(provider)
                self._record_model_metric(provider, model, success=False, latency=0.0)
                errors.append(f"{provider}/{model} attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                await asyncio.sleep(self._retry_delay(attempt))

        # fallback
        try:
            res = await self._execute_query_async("local", "llama3", prompt, system)
            if res and res.strip():
                self._mark_success("local")
                self._record_model_metric("local", "llama3", success=True, latency=0.0)
                self._last_usage = self._estimate_usage(prompt, res)
                self._last_error = {
                    "type": "fallback_used",
                    "provider": start_provider,
                    "model": start_model,
                    "fallback_provider": "local",
                    "fallback_model": "llama3",
                    "errors": errors[-3:],
                }
                return res
        except Exception as exc:
            self._mark_failure("local")
            self._record_model_metric("local", "llama3", success=False, latency=0.0)
            errors.append(f"local/llama3 fallback: {type(exc).__name__}: {exc}")

        self._last_usage = None
        self._last_error = {
            "type": "all_failed",
            "provider": start_provider,
            "model": start_model,
            "errors": errors[-5:],
        }
        return "[API Error: All fallbacks failed or empty response returned.]"

    def query_model(self, provider: str, model: str, prompt: str, system: str = "") -> str:
        return asyncio.run(self.query_model_async(provider, model, prompt, system))

    async def _execute_query_async(self, provider: str, model: str, prompt: str, system: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute_query, provider, model, prompt, system)

    def _execute_query(self, provider: str, model: str, prompt: str, system: str) -> str:
        headers = {"Content-Type": "application/json"}
        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "Athena"
        else:
            url = f"{LLM_API_BASE}/chat/completions"

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_for_provider(provider)) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]

    def query(self, prompt: str, system: str = "") -> str:
        p, m = self.select_best_model("strong")
        return self.query_model(p, m, prompt, system)

    async def query_async(self, prompt: str, system: str = "") -> str:
        p, m = self.select_best_model("strong")
        return await self.query_model_async(p, m, prompt, system)

    def _timeout_for_provider(self, provider: str) -> int:
        if provider == "groq":
            return LLM_GROQ_TIMEOUT_SECONDS
        if provider == "openrouter":
            return LLM_OPENROUTER_TIMEOUT_SECONDS
        return LLM_LOCAL_TIMEOUT_SECONDS

    def _retry_delay(self, attempt: int, rate_limited: bool = False) -> float:
        base = LLM_BASE_RETRY_DELAY_SECONDS * (2 ** attempt)
        if rate_limited:
            base *= 1.8
        return base + random.uniform(0.0, 0.25)

    def _is_circuit_open(self, provider: str) -> bool:
        state = self._circuit_state[provider]
        if state["failures"] < LLM_CIRCUIT_FAILURE_THRESHOLD:
            return False
        return (time.time() - state["opened_at"]) < LLM_CIRCUIT_RESET_SECONDS

    def _mark_failure(self, provider: str) -> None:
        state = self._circuit_state[provider]
        state["failures"] += 1.0
        if state["failures"] >= LLM_CIRCUIT_FAILURE_THRESHOLD:
            state["opened_at"] = time.time()

    def _mark_success(self, provider: str) -> None:
        state = self._circuit_state[provider]
        state["failures"] = 0.0
        state["opened_at"] = 0.0

    def _record_model_metric(self, provider: str, model: str, success: bool, latency: float) -> None:
        key = f"{provider}/{model}"
        perf = self._model_perf.setdefault(
            key,
            {"calls": 0.0, "failures": 0.0, "avg_latency": 0.0},
        )
        perf["calls"] += 1.0
        if not success:
            perf["failures"] += 1.0
        # EMA latency for routing stability
        if latency > 0:
            if perf["avg_latency"] <= 0:
                perf["avg_latency"] = latency
            else:
                perf["avg_latency"] = (0.8 * perf["avg_latency"]) + (0.2 * latency)

    def _estimate_usage(self, prompt: str, completion: str) -> LLMUsage:
        # Approximation; useful for internal routing and trend tracking.
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)
        total_tokens = prompt_tokens + completion_tokens
        input_cost = (prompt_tokens / 1000.0) * DEFAULT_INPUT_TOKEN_COST_PER_1K
        output_cost = (completion_tokens / 1000.0) * DEFAULT_OUTPUT_TOKEN_COST_PER_1K
        total_cost = input_cost + output_cost
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
        )

    def consume_last_error(self) -> Optional[Dict[str, Any]]:
        err = self._last_error
        self._last_error = None
        return err

    def consume_last_usage(self) -> Optional[Dict[str, Any]]:
        usage = self._last_usage
        self._last_usage = None
        if usage is None:
            return None
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "input_cost": round(usage.input_cost, 8),
            "output_cost": round(usage.output_cost, 8),
            "total_cost": round(usage.total_cost, 8),
        }

    def get_embedding(self, text: str) -> List[float]:
        try:
            url = f"{LLM_API_BASE.replace('/v1', '')}/api/embeddings"
            data = {"model": "nomic-embed-text", "prompt": text[:2000]}
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("embedding", [])
        except Exception:
            import hashlib

            h = hashlib.md5(text.encode()).digest()
            return [float(b) / 255.0 for b in h[:16]]
