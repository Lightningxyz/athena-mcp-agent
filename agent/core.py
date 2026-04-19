import asyncio
import os
import time
from typing import Dict, Any

from agent.context_manager import ContextManager
from agent.output_schema import parse_answer_schema, build_repair_prompt
from agent.session_memory import SessionMemory
from llm.client import LLMClient
from mcp_server.tools import list_files
from utils.logger import TraceLogger


class Agent:
    def __init__(self, spinner=None):
        self.llm = LLMClient()
        self.context_manager = ContextManager(self.llm)
        self.logger = TraceLogger(spinner=spinner)

    def _drain_llm_meta(self) -> Dict[str, Any]:
        warning = self.llm.consume_last_error()
        usage = self.llm.consume_last_usage()
        if warning:
            self.logger.log("llm_warning", warning)
        if usage:
            self.logger.log("llm_usage", usage)
        return usage or {}

    async def _query(self, provider: str, model: str, prompt: str, system: str, usage_acc: Dict[str, float]) -> str:
        text = await self.llm.query_model_async(provider, model, prompt, system)
        usage = self._drain_llm_meta()
        usage_acc["prompt_tokens"] += float(usage.get("prompt_tokens", 0))
        usage_acc["completion_tokens"] += float(usage.get("completion_tokens", 0))
        usage_acc["total_tokens"] += float(usage.get("total_tokens", 0))
        usage_acc["total_cost"] += float(usage.get("total_cost", 0.0))
        return text

    async def run_async(self, task_type: str, query: str, path: str) -> str:
        overall_t0 = time.perf_counter()
        self.logger.log("start_task", {"type": task_type, "query": query, "path": path})
        workspace = os.path.abspath(path)
        session_memory = SessionMemory(workspace)
        memory_summary = session_memory.summarize_recent(5)

        fast_provider, fast_model = self.llm.route_model("compression", quality="speed")
        strong_provider, strong_model = self.llm.route_model("answer", quality="quality")
        self.logger.log(
            "model_routing",
            {"fast_pipeline": f"{fast_provider}/{fast_model}", "strong_pipeline": f"{strong_provider}/{strong_model}"},
        )

        usage_acc = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "total_cost": 0.0}
        llm_time = 0.0

        t0_scan = time.perf_counter()
        files, scan_stats = list_files(path)
        t_scan = time.perf_counter() - t0_scan
        self.logger.log(
            "tool_call",
            {"action": "list_files", "files_found": len(files), "scan_time": round(t_scan, 3), "scan_stats": scan_stats},
        )

        extracted_chunks, context_trace = self.context_manager.rank_and_extract(files, query)
        self.logger.log("context_ranking", context_trace)
        intent = context_trace.get("query_intent", "search")
        no_context_found = len(extracted_chunks) == 0
        total_tokens = context_trace.get("estimated_tokens_used", 0)
        context_completeness = min(1.0, total_tokens / 8000.0) if not no_context_found else 0.0

        if no_context_found:
            self.logger.log("fallback", {"reason": "No relevant files found."})
            context_str = "No relevant code context found."
            skip_early = True
        else:
            context_str = "\n\n".join(item["content"] for item in extracted_chunks)
            skip_early = (total_tokens < 600 or intent == "search")

        if skip_early:
            self.logger.log("planning_and_compression", {"action": "skipped", "reason": "small context or direct search intent"})
            reduced_context_str = context_str
        else:
            compression_prompt = (
                f"Query: {query}\n\nContext:\n{context_str}\n\n"
                f"Session Memory:\n{memory_summary}\n\n"
                "Extract the top 3-5 most critical facts from this context that directly answer the query. "
                "Keep them strictly factual and concise."
            )
            self.logger.log("llm_call", {"action": "fast_model_compression", "provider": fast_provider, "model": fast_model})
            t0_llm_comp = time.perf_counter()
            compressed_facts = await self._query(
                fast_provider,
                fast_model,
                compression_prompt,
                "You are a fast, precise code context analyzer.",
                usage_acc,
            )
            llm_time += time.perf_counter() - t0_llm_comp

            if len(compressed_facts.strip()) < 50:
                fallback_chunks = [item["content"] for item in extracted_chunks[:3]]
                reduced_context_str = "\n\n".join(fallback_chunks)
            else:
                top_chunks = [item["content"] for item in extracted_chunks[:2]]
                reduced_context_str = f"EXTRACTED FACTS:\n{compressed_facts}\n\nRAW CRITICAL FILES:\n" + "\n\n".join(top_chunks)

        final_prompt = (
            f"Original Query: {query}\n\n"
            f"Context Payload:\n{reduced_context_str}\n\n"
            f"Session Memory:\n{memory_summary}\n\n"
            "INSTRUCTIONS:\n"
            "Output exactly:\n"
            "Answer:\n<direct answer>\n\n"
            "Justification:\n- <reason>\n- <reason>\n"
            "All claims must be grounded in context. If missing context, say 'insufficient context'."
        )
        self.logger.log("llm_call", {"action": "strong_model_answer", "provider": strong_provider, "model": strong_model})
        t0_llm_ans = time.perf_counter()
        final_answer = await self._query(
            strong_provider,
            strong_model,
            final_prompt,
            "You are Athena, a deterministic AI systems engineer.",
            usage_acc,
        )
        llm_time += time.perf_counter() - t0_llm_ans

        refinement_status = "Skipped"
        confidence_signal = "Low"
        schema_valid = parse_answer_schema(final_answer) is not None
        is_partial = False
        is_rejected = not schema_valid
        last_eval_upper = ""

        for pass_num in range(2):
            eval_prompt = (
                f"Original Query: {query}\n"
                f"Context Payload:\n{reduced_context_str}\n"
                f"Proposed Answer:\n{final_answer}\n\n"
                "Evaluate for hallucination and correctness.\n"
                "Return: APPROVE: ..., PARTIAL: ..., or REJECT: ..."
            )
            self.logger.log("llm_call", {"action": f"self_evaluation_pass_{pass_num + 1}", "provider": strong_provider, "model": strong_model})
            t0_eval = time.perf_counter()
            eval_result = await self._query(
                strong_provider,
                strong_model,
                eval_prompt,
                "You are a strict deterministic evaluator.",
                usage_acc,
            )
            llm_time += time.perf_counter() - t0_eval
            eval_upper = eval_result.strip().upper()
            last_eval_upper = eval_upper
            is_partial = eval_upper.startswith("PARTIAL")
            is_rejected = eval_upper.startswith("REJECT") or not schema_valid

            if eval_upper.startswith("APPROVE") and schema_valid:
                break

            refinement_status = f"Refined ({pass_num + 1} passes)"
            critique = "schema violation" if not schema_valid else eval_result
            self.logger.log("self_correction", {"reason": critique})
            refine_prompt = (
                f"Original Query: {query}\n\n"
                f"Context Payload:\n{reduced_context_str}\n\n"
                f"Previous Attempt:\n{final_answer}\n\n"
                f"Criticism: {critique}\n\n"
                "Fix deterministically and preserve required schema."
            )
            t0_ref = time.perf_counter()
            final_answer = await self._query(
                strong_provider,
                strong_model,
                refine_prompt,
                "You are Athena, a precise AI refiner.",
                usage_acc,
            )
            llm_time += time.perf_counter() - t0_ref
            schema_valid = parse_answer_schema(final_answer) is not None

            if not schema_valid:
                repair_prompt = build_repair_prompt(query, reduced_context_str, final_answer)
                self.logger.log("llm_call", {"action": f"schema_repair_pass_{pass_num + 1}", "provider": strong_provider, "model": strong_model})
                t0_rep = time.perf_counter()
                final_answer = await self._query(
                    strong_provider,
                    strong_model,
                    repair_prompt,
                    "You are a schema repair assistant.",
                    usage_acc,
                )
                llm_time += time.perf_counter() - t0_rep
                schema_valid = parse_answer_schema(final_answer) is not None

        if not schema_valid:
            # last deterministic attempt
            repair_prompt = build_repair_prompt(query, reduced_context_str, final_answer)
            self.logger.log("llm_call", {"action": "schema_repair_final", "provider": strong_provider, "model": strong_model})
            t0_rep2 = time.perf_counter()
            final_answer = await self._query(
                strong_provider,
                strong_model,
                repair_prompt,
                "You are a schema repair assistant.",
                usage_acc,
            )
            llm_time += time.perf_counter() - t0_rep2
            schema_valid = parse_answer_schema(final_answer) is not None

        if schema_valid and refinement_status != "Skipped":
            final_eval_prompt = (
                f"Original Query: {query}\n"
                f"Context Payload:\n{reduced_context_str}\n"
                f"Proposed Answer:\n{final_answer}\n\n"
                "Evaluate for hallucination and correctness.\n"
                "Return: APPROVE: ..., PARTIAL: ..., or REJECT: ..."
            )
            self.logger.log("llm_call", {"action": "self_evaluation_final", "provider": strong_provider, "model": strong_model})
            t0_eval_final = time.perf_counter()
            final_eval_result = await self._query(
                strong_provider,
                strong_model,
                final_eval_prompt,
                "You are a strict deterministic evaluator.",
                usage_acc,
            )
            llm_time += time.perf_counter() - t0_eval_final
            last_eval_upper = final_eval_result.strip().upper()
            is_partial = last_eval_upper.startswith("PARTIAL")
            is_rejected = last_eval_upper.startswith("REJECT")

        if no_context_found or "insufficient context" in final_answer.lower():
            confidence_signal = "Low"
        elif not schema_valid:
            confidence_signal = "Low"
        elif is_rejected:
            confidence_signal = "Low"
        elif is_partial or last_eval_upper.startswith("PARTIAL"):
            confidence_signal = "Medium"
        else:
            confidence_signal = "High" if context_completeness > 0.3 else "Medium"

        total_time = time.perf_counter() - overall_t0
        self.logger.log(
            "performance_metrics",
            {
                "file_scanning_time": round(t_scan, 3),
                "context_processing_time": context_trace.get("execution_times", {}),
                "llm_calls_total_time": round(llm_time, 3),
                "total_runtime": round(total_time, 3),
                "llm_prompt_tokens": int(usage_acc["prompt_tokens"]),
                "llm_completion_tokens": int(usage_acc["completion_tokens"]),
                "llm_total_tokens": int(usage_acc["total_tokens"]),
                "estimated_total_cost": round(usage_acc["total_cost"], 8),
            },
        )
        self.logger.log(
            "evaluation_results",
            {
                "refinement_status": refinement_status,
                "final_confidence": confidence_signal,
                "schema_valid": schema_valid,
            },
        )
        self.logger.log("task_completed", {"final_answer_length": len(final_answer)})
        session_memory.add_entry(
            query=query,
            summary=(final_answer[:200] + ("..." if len(final_answer) > 200 else "")),
            status="completed",
            files_touched=[x.get("file", "") for x in context_trace.get("selected_files", []) if x.get("file")],
            confidence=confidence_signal.lower(),
        )
        return final_answer

    def run(self, task_type: str, query: str, path: str) -> str:
        return asyncio.run(self.run_async(task_type, query, path))
