import asyncio
import json
import os
import shlex
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple

from agent.core import Agent
from agent.patch_engine import PatchEngine, PatchPlan, parse_patch_plan
from agent.session_memory import SessionMemory
from config import AUTOFIX_MAX_ITERATIONS, GIT_AUTO_COMMIT, AUTOFIX_PLAN_CANDIDATES
from config import RUN_ARTIFACTS_DIR
from mcp_server.tools import list_files
from utils.git_ops import GitOps
from utils.safety import validate_command_safety, requires_high_risk_approval


def parse_autofix_plan(raw: str) -> Optional[PatchPlan]:
    # Backward-compatible alias for older callers/tests.
    return parse_patch_plan(raw)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


class AutoFixAgent:
    def __init__(self, base_agent: Agent):
        self.agent = base_agent

    def _build_context(self, query: str, path: str) -> Tuple[str, Dict[str, Any]]:
        files, _scan_stats = list_files(path)
        chunks, trace = self.agent.context_manager.rank_and_extract(files, query)
        if not chunks:
            return "No relevant code context found.", trace
        context_str = "\n\n".join(item["content"] for item in chunks[:5])
        if len(context_str) > 25000:
            context_str = context_str[:25000] + "\n...[truncated]..."
        return context_str, trace

    def _detect_validation_commands(self, workspace: str) -> List[str]:
        cmds: List[str] = []
        has = lambda name: os.path.exists(os.path.join(workspace, name))
        has_any = lambda names: any(os.path.exists(os.path.join(workspace, n)) for n in names)

        if has_any(["pyproject.toml", "setup.py", "setup.cfg", "pytest.ini"]) or os.path.isdir(os.path.join(workspace, "tests")):
            cmds.append("python3 -m unittest discover -s tests -p test*.py -v")
        if has("package.json"):
            cmds.append("npm test")
        if has("go.mod"):
            cmds.append("go test ./...")
        if has("Cargo.toml"):
            cmds.append("cargo test")
        return cmds

    def _strategy_hints(self) -> List[str]:
        return [
            "Minimal surgical fix with the fewest lines changed.",
            "Test-first fix that prioritizes deterministic validations.",
            "Root-cause fix that addresses the underlying defect.",
            "Robustness fix that adds guardrails and edge-case handling.",
            "Refactor-safe fix that improves clarity without broad rewrites.",
        ]

    def _plan_prompt(
        self,
        query: str,
        context_payload: str,
        feedback: str,
        memory_summary: str,
        strategy_hint: str,
    ) -> str:
        return (
            "You are an autonomous software engineering agent.\n"
            "Produce the next best patch plan as strict JSON.\n\n"
            f"Goal:\n{query}\n\n"
            f"Relevant code context:\n{context_payload}\n\n"
            f"Session memory:\n{memory_summary}\n\n"
            f"Feedback from previous iteration:\n{feedback}\n\n"
            f"Strategy hint:\n{strategy_hint}\n\n"
            "Return ONLY valid JSON with schema:\n"
            "{\n"
            '  "summary": "short summary",\n'
            '  "done": false,\n'
            '  "risk": "low|medium|high",\n'
            '  "commit_message": "athena: concise commit message",\n'
            '  "operations": [\n'
            '    {"type":"replace_snippet","file":"path.py","find":"exact text","replace":"new text","reason":"why"},\n'
            '    {"type":"replace_symbol","file":"path.py","symbol":"function_name","symbol_kind":"function|class","new_code":"full replacement code","reason":"why"},\n'
            '    {"type":"append_snippet","file":"path.py","find":"optional anchor text","new_code":"snippet to append","reason":"why"},\n'
            '    {"type":"create_file","file":"new_file.py","new_code":"full file content","reason":"why"},\n'
            '    {"type":"delete_file","file":"obsolete.py","reason":"why"}\n'
            "  ],\n"
            '  "validation_commands": ["python3 -m unittest discover -s tests -p test*.py -v"]\n'
            "}\n"
            "Rules:\n"
            "- Keep operations minimal and deterministic.\n"
            "- Prefer replace_symbol for Python semantic edits.\n"
            "- Use create_file/delete_file only when required by the goal.\n"
            "- Set done=true when no further code edits are required.\n"
            "- Never include markdown fences.\n"
        )

    async def _create_plan(
        self,
        query: str,
        context_payload: str,
        feedback: str,
        memory_summary: str,
        strategy_hint: str = "Minimal surgical fix with deterministic behavior.",
    ) -> Optional[PatchPlan]:
        provider, model = self.agent.llm.route_model("plan", quality="quality")
        prompt = self._plan_prompt(query, context_payload, feedback, memory_summary, strategy_hint)
        raw = await self.agent.llm.query_model_async(
            provider,
            model,
            prompt,
            "You are a deterministic coding planner that outputs strict JSON only.",
        )
        llm_warning = self.agent.llm.consume_last_error()
        llm_usage = self.agent.llm.consume_last_usage()
        if llm_warning:
            self.agent.logger.log("llm_warning", llm_warning)
        if llm_usage:
            self.agent.logger.log("llm_usage", llm_usage)
        return parse_patch_plan(raw)

    def _heuristic_plan_score(self, plan: PatchPlan, preview: Dict[str, Any]) -> float:
        score = 0.0
        risk = (plan.risk or "").lower()
        if risk == "low":
            score += 30.0
        elif risk == "medium":
            score += 15.0
        elif risk == "high":
            score -= 15.0

        op_count = len(plan.operations)
        if plan.done and op_count == 0:
            score += 25.0
        elif op_count > 0:
            score += max(0.0, 24.0 - (2.5 * float(op_count)))
        else:
            score -= 20.0

        if preview.get("ok"):
            score += 35.0
            score -= float(preview.get("failed_count", 0)) * 8.0
        else:
            score -= 50.0

        validation_count = len(plan.validation_commands)
        score += min(10.0, float(validation_count) * 2.0)
        return score

    async def _llm_plan_score(
        self,
        query: str,
        feedback: str,
        plan: PatchPlan,
        preview: Dict[str, Any],
    ) -> Tuple[float, str]:
        provider, model = self.agent.llm.route_model("eval", quality="quality")
        preview_summary = {
            "ok": preview.get("ok", False),
            "failed_count": preview.get("failed_count", 0),
            "applied_files": preview.get("applied_files", []),
        }
        plan_dict = {
            "summary": plan.summary,
            "done": plan.done,
            "risk": plan.risk,
            "operation_count": len(plan.operations),
            "operations": [op.__dict__ for op in plan.operations],
            "validation_commands": plan.validation_commands,
            "commit_message": plan.commit_message,
        }
        prompt = (
            "Score this patch plan quality from 0 to 100.\n"
            "Higher is better if it is safe, likely correct, and minimal.\n\n"
            f"Goal:\n{query}\n\n"
            f"Feedback from prior attempt:\n{feedback}\n\n"
            f"Plan:\n{json.dumps(plan_dict)}\n\n"
            f"Preview summary:\n{json.dumps(preview_summary)}\n\n"
            "Return ONLY JSON:\n"
            '{"score": 0, "reason": "short reason"}'
        )
        raw = await self.agent.llm.query_model_async(
            provider,
            model,
            prompt,
            "You are a deterministic plan evaluator.",
        )
        llm_warning = self.agent.llm.consume_last_error()
        llm_usage = self.agent.llm.consume_last_usage()
        if llm_warning:
            self.agent.logger.log("llm_warning", llm_warning)
        if llm_usage:
            self.agent.logger.log("llm_usage", llm_usage)

        parsed = _extract_json_object(raw or "")
        if not parsed:
            return 45.0, "llm score parse failed"
        try:
            score = float(parsed.get("score", 45))
        except Exception:
            score = 45.0
        score = max(0.0, min(100.0, score))
        reason = str(parsed.get("reason", "no reason")).strip()
        return score, reason

    async def _create_best_plan(
        self,
        query: str,
        context_payload: str,
        feedback: str,
        memory_summary: str,
        patch_engine: PatchEngine,
        plan_candidates: int,
    ) -> Tuple[Optional[PatchPlan], Dict[str, Any], List[Dict[str, Any]]]:
        hints = self._strategy_hints()
        candidates = max(1, plan_candidates)
        tasks = []
        hint_used = []
        for i in range(candidates):
            hint = hints[i % len(hints)]
            hint_used.append(hint)
            try:
                task = self._create_plan(query, context_payload, feedback, memory_summary, hint)
            except TypeError:
                # Backward compatibility for monkeypatched _create_plan in tests/extensions.
                task = self._create_plan(query, context_payload, feedback, memory_summary)
            tasks.append(task)
        plans = await asyncio.gather(*tasks)

        plan_infos: List[Dict[str, Any]] = []
        llm_score_tasks = []
        llm_score_idx = []
        for idx, plan in enumerate(plans):
            info: Dict[str, Any] = {
                "candidate_index": idx,
                "strategy_hint": hint_used[idx],
                "valid_plan": plan is not None,
            }
            if plan is None:
                info["selection_score"] = -999.0
                info["reason"] = "plan parse failure"
                plan_infos.append(info)
                continue
            preview = patch_engine.apply_plan(plan, dry_run=True)
            heuristic = self._heuristic_plan_score(plan, preview)
            info.update(
                {
                    "summary": plan.summary,
                    "risk": plan.risk,
                    "done": plan.done,
                    "operation_count": len(plan.operations),
                    "preview_ok": bool(preview.get("ok")),
                    "preview_failed_count": int(preview.get("failed_count", 0)),
                    "heuristic_score": round(heuristic, 2),
                    "_plan_ref": plan,
                    "_preview_ref": preview,
                }
            )
            llm_score_tasks.append(self._llm_plan_score(query, feedback, plan, preview))
            llm_score_idx.append(idx)
            plan_infos.append(info)

        if llm_score_tasks:
            llm_scores = await asyncio.gather(*llm_score_tasks)
            mapping = {idx: llm_scores[pos] for pos, idx in enumerate(llm_score_idx)}
        else:
            mapping = {}

        best_plan = None
        best_preview = {"ok": False, "failed_count": 9999}
        best_score = float("-inf")
        for info in plan_infos:
            idx = info["candidate_index"]
            llm_score, llm_reason = mapping.get(idx, (40.0, "no llm score"))
            heuristic = float(info.get("heuristic_score", -999.0))
            selection = heuristic + (0.6 * llm_score)
            info["llm_score"] = round(llm_score, 2)
            info["llm_reason"] = llm_reason
            info["selection_score"] = round(selection, 2)
            if selection > best_score and "_plan_ref" in info:
                best_score = selection
                best_plan = info["_plan_ref"]
                best_preview = info["_preview_ref"]

        # cleanup private refs before report
        for info in plan_infos:
            info.pop("_plan_ref", None)
            info.pop("_preview_ref", None)

        return best_plan, best_preview, sorted(plan_infos, key=lambda x: x.get("selection_score", -9999), reverse=True)

    def _run_validation(self, workspace: str, commands: List[str]) -> List[Dict[str, Any]]:
        results = []
        for cmd in commands[:8]:
            safe, reason = validate_command_safety(cmd)
            if not safe:
                results.append({"command": cmd, "returncode": 126, "output": f"blocked by safety policy: {reason}"})
                continue
            try:
                args = shlex.split(cmd)
                proc = subprocess.run(
                    args,
                    cwd=workspace,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
                if len(output) > 3000:
                    output = output[:3000] + "\n...[truncated]..."
                results.append({"command": cmd, "returncode": proc.returncode, "output": output})
            except Exception as exc:
                results.append({"command": cmd, "returncode": 1, "output": f"validation error: {exc}"})
        return results

    def _write_artifact(self, workspace: str, report: Dict[str, Any]) -> Optional[str]:
        artifacts_dir = RUN_ARTIFACTS_DIR if os.path.isabs(RUN_ARTIFACTS_DIR) else os.path.join(workspace, RUN_ARTIFACTS_DIR)
        os.makedirs(artifacts_dir, exist_ok=True)
        filename = f"autofix-{time.strftime('%Y%m%d-%H%M%S')}.json"
        target = os.path.join(artifacts_dir, filename)
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return target
        except Exception:
            return None

    async def run_async(
        self,
        query: str,
        path: str,
        apply: bool,
        max_iterations: int = AUTOFIX_MAX_ITERATIONS,
        plan_candidates: int = AUTOFIX_PLAN_CANDIDATES,
        validation_commands: Optional[List[str]] = None,
        approve_high_risk: bool = False,
        git_mode: bool = False,
        create_branch: bool = False,
        auto_commit: bool = GIT_AUTO_COMMIT,
    ) -> Dict[str, Any]:
        workspace = os.path.abspath(path)
        patch_engine = PatchEngine(workspace)
        memory = SessionMemory(workspace)
        git_ops = GitOps(workspace)

        context_payload, retrieval_trace = self._build_context(query, workspace)
        self.agent.logger.log(
            "autofix_start",
            {
                "query": query,
                "path": workspace,
                "max_iterations": max_iterations,
                "plan_candidates": plan_candidates,
                "apply": apply,
                "git_mode": git_mode,
            },
        )
        run_report: Dict[str, Any] = {
            "query": query,
            "path": workspace,
            "started_at": time.time(),
            "retrieval_trace": retrieval_trace,
            "iterations": [],
            "git": {},
        }

        if git_mode:
            if not git_ops.is_repo():
                run_report["git"] = {"enabled": True, "active": False, "reason": "not a git repository"}
            else:
                run_report["git"] = {
                    "enabled": True,
                    "active": True,
                    "branch_before": git_ops.current_branch(),
                    "dirty_before": git_ops.working_tree_dirty(),
                    "head_before": git_ops.last_commit(),
                }
                if create_branch:
                    suffix = query[:40]
                    branch_res = git_ops.create_work_branch(suffix_hint=suffix)
                    run_report["git"]["branch_create"] = branch_res

        feedback = "No previous iteration."
        summary_for_memory = "No execution."
        status_for_memory = "unknown"
        files_touched: List[str] = []
        confidence_for_memory = "unknown"

        for idx in range(max(1, max_iterations)):
            memory_summary = memory.summarize_recent(6)
            plan, preview, candidate_info = await self._create_best_plan(
                query=query,
                context_payload=context_payload,
                feedback=feedback,
                memory_summary=memory_summary,
                patch_engine=patch_engine,
                plan_candidates=max(1, plan_candidates),
            )
            if plan is None:
                run_report["iterations"].append(
                    {
                        "iteration": idx + 1,
                        "error": "planner did not return valid JSON",
                        "candidates": candidate_info,
                        "done": True,
                    }
                )
                status_for_memory = "planner_error"
                summary_for_memory = "Planner response was not parseable JSON."
                confidence_for_memory = "low"
                break

            iter_report: Dict[str, Any] = {
                "iteration": idx + 1,
                "summary": plan.summary,
                "risk": plan.risk,
                "done": plan.done,
                "operation_count": len(plan.operations),
                "operations": [op.__dict__ for op in plan.operations],
                "commit_message": plan.commit_message,
                "candidate_plans": candidate_info,
            }
            self.agent.logger.log(
                "autofix_plan",
                {
                    "iteration": idx + 1,
                    "risk": plan.risk,
                    "done": plan.done,
                    "operation_count": len(plan.operations),
                    "candidate_count": len(candidate_info),
                },
            )

            needs_approval, reason = requires_high_risk_approval(plan.risk, approve_high_risk)
            if needs_approval and apply:
                iter_report["blocked"] = reason
                run_report["iterations"].append(iter_report)
                status_for_memory = "blocked"
                summary_for_memory = reason
                confidence_for_memory = "low"
                break

            if plan.done or not plan.operations:
                iter_report["note"] = "planner marked task complete"
                run_report["iterations"].append(iter_report)
                status_for_memory = "completed"
                summary_for_memory = plan.summary
                confidence_for_memory = "medium"
                break

            iter_report["preview"] = preview
            if not preview["ok"]:
                feedback = "Patch preview failed:\n" + "\n".join(
                    f"- {r.get('file')}: {r.get('reason')}" for r in preview["results"] if not r.get("ok")
                )
                iter_report["feedback"] = feedback
                run_report["iterations"].append(iter_report)
                status_for_memory = "preview_failed"
                summary_for_memory = "Patch preview failed due to non-deterministic operations."
                confidence_for_memory = "low"
                continue

            if apply:
                checkpoint = patch_engine.create_checkpoint(
                    [op.file for op in plan.operations],
                    reason=f"autofix iteration {idx+1}: {plan.summary}",
                )
                iter_report["checkpoint"] = checkpoint
                apply_result = patch_engine.apply_plan(plan, dry_run=False)
                iter_report["apply"] = apply_result
                self.agent.logger.log(
                    "autofix_apply",
                    {
                        "iteration": idx + 1,
                        "ok": apply_result["ok"],
                        "failed_count": apply_result["failed_count"],
                        "applied_files": len(apply_result["applied_files"]),
                    },
                )
                if not apply_result["ok"]:
                    rollback_result = patch_engine.rollback()
                    iter_report["rollback"] = rollback_result
                    feedback = "Apply failed and rollback executed."
                    run_report["iterations"].append(iter_report)
                    status_for_memory = "apply_failed"
                    summary_for_memory = "Patch application failed and was rolled back."
                    confidence_for_memory = "low"
                    break
                files_touched.extend(apply_result["applied_files"])
            else:
                iter_report["apply"] = {"ok": True, "dry_run": True}
                feedback = "Dry run only. No file changes were written."
                run_report["iterations"].append(iter_report)
                status_for_memory = "preview_only"
                summary_for_memory = plan.summary
                confidence_for_memory = "medium"
                continue

            commands = validation_commands if validation_commands else plan.validation_commands
            if not commands:
                commands = self._detect_validation_commands(workspace)
                if commands:
                    iter_report["validation_auto_detected"] = commands
            if commands:
                validation = self._run_validation(workspace, commands)
                iter_report["validation"] = validation
                failed = [x for x in validation if x["returncode"] != 0]
                if failed:
                    if checkpoint.get("ok"):
                        iter_report["checkpoint_restore"] = patch_engine.restore_checkpoint(checkpoint["checkpoint_id"])
                    feedback = "Validation failed:\n" + "\n\n".join(f"[{x['command']}]\n{x['output']}" for x in failed)
                    status_for_memory = "validation_failed"
                    confidence_for_memory = "low"
                else:
                    feedback = "Validation passed."
                    status_for_memory = "validated"
                    confidence_for_memory = "high"
            else:
                feedback = "No validation commands provided."
                status_for_memory = "applied_no_validation"
                confidence_for_memory = "medium"

            if git_mode and run_report.get("git", {}).get("active"):
                diff_stat = git_ops.diff_stat()
                iter_report["git_diff_stat"] = diff_stat
                if auto_commit and diff_stat.strip():
                    add_res = git_ops.add_all()
                    commit_res = git_ops.commit(plan.commit_message)
                    iter_report["git_commit"] = {"add": add_res, "commit": commit_res}

            run_report["iterations"].append(iter_report)
            summary_for_memory = plan.summary

            if status_for_memory in {"validated", "completed"}:
                break

        run_report["ended_at"] = time.time()
        run_report["status"] = status_for_memory or "completed"
        run_report["files_touched"] = sorted(set(files_touched))
        if git_mode and run_report.get("git", {}).get("active"):
            run_report["git"]["branch_after"] = git_ops.current_branch()
            run_report["git"]["head_after"] = git_ops.last_commit()
            if run_report["files_touched"]:
                bullet_files = "\n".join(f"- {f}" for f in run_report["files_touched"][:20])
            else:
                bullet_files = "- (no files touched)"
            run_report["git"]["pr_draft"] = (
                f"## Summary\n{summary_for_memory}\n\n"
                f"## Goal\n{query}\n\n"
                "## Changes\n"
                f"{bullet_files}\n\n"
                "## Validation\n"
                "See iteration validation logs in the autofix report."
            )
        memory.add_entry(
            query=query,
            summary=summary_for_memory,
            status=run_report["status"],
            files_touched=run_report["files_touched"],
            confidence=confidence_for_memory,
        )
        artifact = self._write_artifact(workspace, run_report)
        if artifact:
            run_report["artifact_file"] = artifact
        return run_report

    def run(
        self,
        query: str,
        path: str,
        apply: bool,
        max_iterations: int = AUTOFIX_MAX_ITERATIONS,
        plan_candidates: int = AUTOFIX_PLAN_CANDIDATES,
        validation_commands: Optional[List[str]] = None,
        approve_high_risk: bool = False,
        git_mode: bool = False,
        create_branch: bool = False,
        auto_commit: bool = GIT_AUTO_COMMIT,
    ) -> Dict[str, Any]:
        return asyncio.run(
            self.run_async(
                query=query,
                path=path,
                apply=apply,
                max_iterations=max_iterations,
                plan_candidates=plan_candidates,
                validation_commands=validation_commands,
                approve_high_risk=approve_high_risk,
                git_mode=git_mode,
                create_branch=create_branch,
                auto_commit=auto_commit,
            )
        )
