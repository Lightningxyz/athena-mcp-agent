import os
import shutil
import subprocess
import time
from typing import Dict, Any, Optional, List

from config import GIT_AUTO_BRANCH_PREFIX


class GitOps:
    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(workspace)
        self.git_available = shutil.which("git") is not None

    def _run(self, args: List[str]) -> Dict[str, Any]:
        if not self.git_available:
            return {"ok": False, "code": 127, "out": "", "err": "git not found"}
        try:
            proc = subprocess.run(
                ["git"] + args,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {"ok": proc.returncode == 0, "code": proc.returncode, "out": proc.stdout.strip(), "err": proc.stderr.strip()}
        except Exception as exc:
            return {"ok": False, "code": 1, "out": "", "err": str(exc)}

    def is_repo(self) -> bool:
        res = self._run(["rev-parse", "--is-inside-work-tree"])
        return bool(res["ok"] and res["out"].strip() == "true")

    def current_branch(self) -> Optional[str]:
        res = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return res["out"] if res["ok"] else None

    def working_tree_dirty(self) -> bool:
        res = self._run(["status", "--porcelain"])
        return bool(res["ok"] and res["out"])

    def create_work_branch(self, suffix_hint: str = "session") -> Dict[str, Any]:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_hint = "".join(c if c.isalnum() or c in "-_" else "-" for c in suffix_hint.lower())[:40] or "work"
        branch = f"{GIT_AUTO_BRANCH_PREFIX}/{safe_hint}-{timestamp}"
        return self._run(["checkout", "-b", branch])

    def diff_stat(self) -> str:
        res = self._run(["diff", "--stat"])
        return res["out"] if res["ok"] else res["err"]

    def staged_or_working_diff(self) -> str:
        res = self._run(["diff"])
        return res["out"] if res["ok"] else res["err"]

    def add_all(self) -> Dict[str, Any]:
        return self._run(["add", "-A"])

    def commit(self, message: str) -> Dict[str, Any]:
        return self._run(["commit", "-m", message])

    def last_commit(self) -> Optional[str]:
        res = self._run(["rev-parse", "HEAD"])
        return res["out"] if res["ok"] else None

    def revert_last_commit(self) -> Dict[str, Any]:
        # Non-destructive rollback via revert, not reset.
        return self._run(["revert", "--no-edit", "HEAD"])
