from typing import Dict, Any, List

def calculate_quality_score(output: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    matches = sum(1 for kw in keywords if kw.lower() in output.lower())
    return matches / len(keywords)

def calculate_efficiency_score(runtime: float, tokens: int, quality: float) -> float:
    norm_time = 10.0 / max(runtime, 0.1)
    norm_tokens = 5000.0 / max(tokens, 1)
    return (norm_time * 0.4) + (norm_tokens * 0.3) + (quality * 5.0)

def extract_metrics_from_trace(trace_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "runtime": 0.0,
        "llm_time": 0.0,
        "tokens": 0,
        "chars": 0,
        "files_used": 0,
        "chunks_used": 0,
        "planning_skipped": False,
        "refinement_triggered": False,
        "schema_valid": False,
        "cost": 0.0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
    }
    
    for event in trace_log:
        t = event.get("event")
        d = event.get("details", {})
        
        if t == "planning_and_compression" and d.get("action") == "skipped":
            metrics["planning_skipped"] = True
            
        elif t == "context_ranking":
            metrics["tokens"] = d.get("estimated_tokens_used", 0)
            metrics["chars"] = d.get("total_chars_used", 0)
            metrics["files_used"] = len(d.get("selected_files", []))
            
        elif t == "performance_metrics":
            metrics["runtime"] = d.get("total_runtime", 0.0)
            metrics["llm_time"] = d.get("llm_calls_total_time", 0.0)
            metrics["cost"] = d.get("estimated_total_cost", 0.0)
            metrics["llm_prompt_tokens"] = d.get("llm_prompt_tokens", 0)
            metrics["llm_completion_tokens"] = d.get("llm_completion_tokens", 0)
            
        elif t == "evaluation_results":
            ref_status = d.get("refinement_status", "Skipped")
            metrics["refinement_triggered"] = ref_status != "Skipped"
            metrics["schema_valid"] = bool(d.get("schema_valid", False))

    return metrics
