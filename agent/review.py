import asyncio
import json
import os
import subprocess
import time
from typing import Dict, Any, List, Optional

from agent.core import Agent
from config import RUN_ARTIFACTS_DIR
from utils.git_ops import GitOps


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _severity_rank(level: str) -> int:
    level = (level or "").lower()
    if level == "critical":
        return 4
    if level == "high":
        return 3
    if level == "medium":
        return 2
    if level == "low":
        return 1
    return 0


class ReviewAgent:
    def __init__(self, base_agent: Agent):
        self.agent = base_agent

    def _collect_diff(self, workspace: str, staged: bool = False, base_ref: Optional[str] = None) -> Dict[str, Any]:
        git = GitOps(workspace)
        if not git.is_repo():
            return {"ok": False, "reason": "not a git repository", "diff": ""}

        cmd = ["git", "diff"]
        if staged:
            cmd = ["git", "diff", "--cached"]
        elif base_ref:
            cmd = ["git", "diff", f"{base_ref}...HEAD"]

        try:
            proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return {"ok": False, "reason": str(exc), "diff": ""}

        diff_text = (proc.stdout or "").strip()
        if not diff_text and not staged and base_ref is None:
            # fallback to staged diff if working tree clean
            proc2 = subprocess.run(["git", "diff", "--cached"], cwd=workspace, capture_output=True, text=True, timeout=30)
            diff_text = (proc2.stdout or "").strip()
        if not diff_text:
            return {"ok": False, "reason": "no diff to review", "diff": ""}
        if len(diff_text) > 50000:
            diff_text = diff_text[:50000] + "\n...[diff truncated]..."
        return {"ok": True, "reason": "ok", "diff": diff_text}

    async def _review_diff(self, diff_text: str, focus: str = "") -> Dict[str, Any]:
        provider, model = self.agent.llm.route_model("eval", quality="quality")
        prompt = (
            "You are a senior staff engineer performing code review.\n"
            "Review the diff for bugs, regressions, missing tests, reliability risks, and security issues.\n"
            "Be strict, concrete, and concise.\n\n"
            f"Focus hint (optional): {focus or 'general'}\n\n"
            f"Diff:\n{diff_text}\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "summary": "short paragraph",\n'
            '  "overall_risk": "low|medium|high|critical",\n'
            '  "findings": [\n'
            '    {"severity":"critical|high|medium|low","file":"path","title":"short title","details":"why this matters","suggested_fix":"concrete fix","line_hint":"optional"}\n'
            "  ],\n"
            '  "missing_tests": ["item1", "item2"]\n'
            "}\n"
            "If no findings, return empty findings and explain residual risks."
        )
        raw = await self.agent.llm.query_model_async(
            provider,
            model,
            prompt,
            "You are a deterministic code reviewer that outputs strict JSON.",
        )
        llm_warning = self.agent.llm.consume_last_error()
        llm_usage = self.agent.llm.consume_last_usage()
        if llm_warning:
            self.agent.logger.log("llm_warning", llm_warning)
        if llm_usage:
            self.agent.logger.log("llm_usage", llm_usage)

        parsed = _extract_json(raw)
        if not parsed:
            return {
                "summary": "Reviewer output could not be parsed as JSON.",
                "overall_risk": "medium",
                "findings": [],
                "missing_tests": [],
                "raw": raw[:2000],
            }
        findings = parsed.get("findings", [])
        if isinstance(findings, list):
            findings = sorted(findings, key=lambda x: _severity_rank(str(x.get("severity", ""))), reverse=True)
            parsed["findings"] = findings
        return parsed

    def _write_artifact(self, workspace: str, report: Dict[str, Any]) -> Optional[str]:
        artifacts_dir = RUN_ARTIFACTS_DIR if os.path.isabs(RUN_ARTIFACTS_DIR) else os.path.join(workspace, RUN_ARTIFACTS_DIR)
        os.makedirs(artifacts_dir, exist_ok=True)
        target = os.path.join(artifacts_dir, f"review-{time.strftime('%Y%m%d-%H%M%S')}.json")
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return target
        except Exception:
            return None

    async def run_async(
        self,
        path: str,
        focus: str = "",
        staged: bool = False,
        base_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        workspace = os.path.abspath(path)
        collected = self._collect_diff(workspace, staged=staged, base_ref=base_ref)
        if not collected["ok"]:
            report = {"path": workspace, "status": "no_diff", "reason": collected["reason"], "findings": []}
            artifact = self._write_artifact(workspace, report)
            if artifact:
                report["artifact_file"] = artifact
            return report

        self.agent.logger.log(
            "review_start",
            {"path": workspace, "staged": staged, "base_ref": base_ref or "", "focus": focus or ""},
        )
        reviewed = await self._review_diff(collected["diff"], focus=focus)
        report = {
            "path": workspace,
            "status": "completed",
            "focus": focus,
            "staged": staged,
            "base_ref": base_ref,
            "summary": reviewed.get("summary", ""),
            "overall_risk": reviewed.get("overall_risk", "medium"),
            "findings": reviewed.get("findings", []),
            "missing_tests": reviewed.get("missing_tests", []),
        }
        self.agent.logger.log(
            "review_result",
            {
                "overall_risk": report["overall_risk"],
                "finding_count": len(report["findings"]),
                "missing_test_count": len(report["missing_tests"]),
            },
        )
        artifact = self._write_artifact(workspace, report)
        if artifact:
            report["artifact_file"] = artifact
        return report

    def run(
        self,
        path: str,
        focus: str = "",
        staged: bool = False,
        base_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        return asyncio.run(self.run_async(path=path, focus=focus, staged=staged, base_ref=base_ref))
