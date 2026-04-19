import json
import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import Agent
from benchmark.metrics import calculate_quality_score, calculate_efficiency_score, extract_metrics_from_trace

TEST_CASES = [
    {"query": "Explain how context_manager works", "path": ".", "expected_keywords": ["tokens", "ranking", "score", "extract"]},
    {"query": "What are the configuration constraints set?", "path": ".", "expected_keywords": ["MAX_CONTEXT_TOKENS", "MAX_FILE_BYTES"]},
    {"query": "List all MCP tools available.", "path": ".", "expected_keywords": ["list_files", "read_file"]},
]

BASELINE_PATH = os.getenv("BENCHMARK_BASELINE_PATH", "benchmark/baseline.json")
MAX_RUNTIME_REGRESSION_PCT = float(os.getenv("BENCHMARK_MAX_RUNTIME_REGRESSION_PCT", "20"))
MAX_COST_REGRESSION_PCT = float(os.getenv("BENCHMARK_MAX_COST_REGRESSION_PCT", "25"))
ALLOW_SCHEMA_FAILURE = os.getenv("BENCHMARK_ALLOW_SCHEMA_FAILURE", "false").lower() == "true"


def mock_no_refinement(agent):
    original_query_async = agent.llm.query_model_async

    async def mocked(provider, model, prompt, system=""):
        if "strict deterministic evaluator" in system.lower():
            return "APPROVE: High Confidence"
        return await original_query_async(provider, model, prompt, system)

    agent.llm.query_model_async = mocked


def mock_fast_only(agent):
    original_route = agent.llm.route_model

    def mocked(_task, quality="balanced"):
        return original_route("fast", quality=quality)

    agent.llm.route_model = mocked


class BenchmarkRunner:
    def execute_test(self, test: Dict[str, Any], config_name: str, modifiers: List[callable] = None) -> Dict[str, Any]:
        agent = Agent()
        if modifiers:
            for mod in modifiers:
                mod(agent)

        result = agent.run("benchmark", test["query"], test["path"])
        metrics = extract_metrics_from_trace(agent.logger.trace)
        quality = calculate_quality_score(result, test.get("expected_keywords", []))
        efficiency = calculate_efficiency_score(metrics["runtime"], metrics["tokens"], quality)
        return {
            "query": test["query"],
            "config": config_name,
            "runtime": metrics["runtime"],
            "llm_time": metrics["llm_time"],
            "tokens": metrics["tokens"],
            "chars": metrics["chars"],
            "files": metrics["files_used"],
            "chunks": metrics["files_used"],
            "planning_skipped": metrics["planning_skipped"],
            "refinement_triggered": metrics["refinement_triggered"],
            "schema_valid": metrics["schema_valid"],
            "cost": metrics["cost"],
            "llm_prompt_tokens": metrics["llm_prompt_tokens"],
            "llm_completion_tokens": metrics["llm_completion_tokens"],
            "quality_score": round(quality, 2),
            "efficiency_score": round(efficiency, 2),
        }

    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        print("\n" + "=" * 40)
        print("=== BENCHMARK RESULTS ===")
        print("=" * 40)
        for r in results:
            print(f"\nQuery: {r['query']}")
            print(f"Config: {r['config']}")
            print(f"Runtime: {r['runtime']:.2f}s (LLM: {r['llm_time']:.2f}s)")
            print(f"Tokens Used: {r['tokens']} ({r['chars']} chars)")
            print(f"LLM Prompt/Completion Tokens: {r['llm_prompt_tokens']}/{r['llm_completion_tokens']}")
            print(f"Estimated Cost: ${r['cost']:.6f}")
            print(f"Files Used: {r['files']}")
            print(f"Schema Valid: {'Yes' if r['schema_valid'] else 'No'}")
            print(f"Refinement: {'Yes' if r['refinement_triggered'] else 'No'}")
            print(f"Quality Score: {r['quality_score']:.2f}/1.0")
            print(f"Efficiency Score: {r['efficiency_score']:.2f}")
            print("-" * 40)

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        count = max(1, len(results))
        return {
            "avg_runtime": sum(r["runtime"] for r in results) / count,
            "avg_cost": sum(r["cost"] for r in results) / count,
            "schema_pass_rate": sum(1 for r in results if r["schema_valid"]) / count,
        }

    def _load_baseline(self) -> Dict[str, float]:
        if not os.path.exists(BASELINE_PATH):
            return {}
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_baseline(self, aggregate: Dict[str, float]) -> None:
        os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)

    def enforce_regression_gates(self, aggregate: Dict[str, float]) -> None:
        baseline = self._load_baseline()
        if not baseline:
            self._save_baseline(aggregate)
            print(f"\nNo baseline found. Saved new baseline to {BASELINE_PATH}")
            return

        runtime_limit = baseline["avg_runtime"] * (1 + (MAX_RUNTIME_REGRESSION_PCT / 100.0))
        cost_limit = baseline["avg_cost"] * (1 + (MAX_COST_REGRESSION_PCT / 100.0))

        failures = []
        if aggregate["avg_runtime"] > runtime_limit:
            failures.append(
                f"Runtime regression: {aggregate['avg_runtime']:.3f}s > allowed {runtime_limit:.3f}s"
            )
        if aggregate["avg_cost"] > cost_limit:
            failures.append(
                f"Cost regression: ${aggregate['avg_cost']:.6f} > allowed ${cost_limit:.6f}"
            )
        if not ALLOW_SCHEMA_FAILURE and aggregate["schema_pass_rate"] < 1.0:
            failures.append(
                f"Schema pass rate regression: {aggregate['schema_pass_rate']:.2%} (expected 100%)"
            )

        if failures:
            raise RuntimeError("Benchmark regression gates failed:\n- " + "\n- ".join(failures))


def main():
    runner = BenchmarkRunner()
    results = []
    print("Running Benchmark Phase 1: Baseline Multi-Model...")
    for t in TEST_CASES:
        results.append(runner.execute_test(t, "Baseline (Multi-Model, Full Pass)"))
    print("Running Benchmark Phase 2: Fast-Model Only...")
    for t in TEST_CASES:
        results.append(runner.execute_test(t, "Fast Only", [mock_fast_only]))
    print("Running Benchmark Phase 3: No Refinement...")
    for t in TEST_CASES:
        results.append(runner.execute_test(t, "No Refine", [mock_no_refinement]))

    runner.print_summary(results)
    aggregate = runner._aggregate(results)
    print("\nAggregate:")
    print(json.dumps(aggregate, indent=2))
    runner.enforce_regression_gates(aggregate)


if __name__ == "__main__":
    main()
