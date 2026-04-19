import json
import os
import sys
import time
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import Agent
from config import EVAL_HISTORY_FILE


def load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_response(response: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / max(1, len(keywords))


def latest_runtime(trace: List[Dict[str, Any]]) -> float:
    for event in reversed(trace):
        if event.get("event") == "performance_metrics":
            return float(event.get("details", {}).get("total_runtime", 0.0))
    return 0.0


def run_eval(tasks_path: str = "eval/tasks.json") -> Dict[str, Any]:
    tasks = load_tasks(tasks_path)
    agent = Agent()
    results = []
    for task in tasks:
        t0 = time.perf_counter()
        out = agent.run("debug", task["query"], task.get("path", "."))
        elapsed = time.perf_counter() - t0
        score = score_response(out, task.get("expected_keywords", []))
        trace_runtime = latest_runtime(agent.logger.trace)
        results.append(
            {
                "name": task["name"],
                "score": round(score, 3),
                "wall_time": round(elapsed, 3),
                "trace_runtime": round(trace_runtime, 3),
            }
        )

    avg_score = sum(r["score"] for r in results) / max(1, len(results))
    avg_runtime = sum(r["trace_runtime"] for r in results) / max(1, len(results))
    summary = {
        "timestamp": time.time(),
        "task_count": len(results),
        "avg_score": round(avg_score, 4),
        "avg_runtime": round(avg_runtime, 4),
        "results": results,
    }
    return summary


def load_history(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(path: str, entries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def enforce_gates(summary: Dict[str, Any], previous: Dict[str, Any]) -> None:
    if not previous:
        return
    # Gate 1: quality should not regress by more than 8 points.
    if summary["avg_score"] < (previous.get("avg_score", 0.0) - 0.08):
        raise RuntimeError(
            f"Eval quality regression: {summary['avg_score']:.3f} vs previous {previous.get('avg_score', 0.0):.3f}"
        )
    # Gate 2: runtime should not grow by more than 30% unless quality improved by >=5 points.
    prev_runtime = max(0.001, float(previous.get("avg_runtime", 0.0)))
    runtime_growth = (summary["avg_runtime"] - prev_runtime) / prev_runtime
    quality_gain = summary["avg_score"] - float(previous.get("avg_score", 0.0))
    if runtime_growth > 0.30 and quality_gain < 0.05:
        raise RuntimeError(
            f"Eval runtime regression: +{runtime_growth*100:.1f}% with insufficient quality gain ({quality_gain:.3f})"
        )


def main() -> int:
    summary = run_eval()
    history = load_history(EVAL_HISTORY_FILE)
    previous = history[-1] if history else {}
    enforce_gates(summary, previous)
    history.append(summary)
    history = history[-50:]
    save_history(EVAL_HISTORY_FILE, history)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
