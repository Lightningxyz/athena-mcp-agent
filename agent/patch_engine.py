import ast
import difflib
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from config import CHECKPOINTS_DIR
from utils.safety import normalize_workspace_path, is_within_workspace


@dataclass
class PatchOperation:
    op_type: str
    file: str
    reason: str
    find: str = ""
    replace: str = ""
    symbol: str = ""
    symbol_kind: str = ""
    new_code: str = ""


@dataclass
class PatchPlan:
    summary: str
    done: bool
    risk: str
    operations: List[PatchOperation]
    validation_commands: List[str]
    commit_message: str


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def parse_patch_plan(raw: str) -> Optional[PatchPlan]:
    payload = _extract_json(raw)
    if not payload:
        return None
    ops: List[PatchOperation] = []
    raw_ops = payload.get("operations", [])
    if isinstance(raw_ops, list):
        for item in raw_ops:
            if not isinstance(item, dict):
                continue
            op_type = str(item.get("type", "")).strip()
            file = str(item.get("file", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not op_type or not file:
                continue
            ops.append(
                PatchOperation(
                    op_type=op_type,
                    file=file,
                    reason=reason,
                    find=str(item.get("find", "")),
                    replace=str(item.get("replace", "")),
                    symbol=str(item.get("symbol", "")),
                    symbol_kind=str(item.get("symbol_kind", "")),
                    new_code=str(item.get("new_code", "")),
                )
            )
    validation_commands = payload.get("validation_commands", [])
    if not isinstance(validation_commands, list):
        validation_commands = []
    validation_commands = [str(x).strip() for x in validation_commands if str(x).strip()]
    return PatchPlan(
        summary=str(payload.get("summary", "")).strip() or "No summary provided.",
        done=bool(payload.get("done", False)),
        risk=str(payload.get("risk", "medium")).strip().lower() or "medium",
        operations=ops,
        validation_commands=validation_commands,
        commit_message=str(payload.get("commit_message", "athena: apply automated patch")).strip()
        or "athena: apply automated patch",
    )


class PatchEngine:
    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(workspace)
        self._backups: Dict[str, str] = {}
        self.checkpoints_dir = CHECKPOINTS_DIR if os.path.isabs(CHECKPOINTS_DIR) else os.path.join(self.workspace, CHECKPOINTS_DIR)

    def _load(self, target_abs: str) -> Tuple[Optional[str], Optional[str]]:
        if not os.path.exists(target_abs):
            return None, "file not found"
        try:
            with open(target_abs, "r", encoding="utf-8") as f:
                return f.read(), None
        except Exception as exc:
            return None, f"read error: {exc}"

    def _save(self, target_abs: str, content: str) -> Optional[str]:
        try:
            with open(target_abs, "w", encoding="utf-8") as f:
                f.write(content)
            return None
        except Exception as exc:
            return f"write error: {exc}"

    def _record_backup(self, target_abs: str, original: str) -> None:
        if target_abs not in self._backups:
            self._backups[target_abs] = original

    def rollback(self) -> Dict[str, Any]:
        restored = []
        errors = []
        for path, original in self._backups.items():
            err = self._save(path, original)
            if err:
                errors.append({"file": path, "reason": err})
            else:
                restored.append(path)
        self._backups.clear()
        return {"restored": restored, "errors": errors}

    def create_checkpoint(self, file_paths: List[str], reason: str = "") -> Dict[str, Any]:
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        checkpoint_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        target_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        os.makedirs(target_dir, exist_ok=True)
        manifest = {"id": checkpoint_id, "created_at": time.time(), "reason": reason, "files": [], "absent_files": []}

        for rel in sorted(set(file_paths)):
            abs_path = normalize_workspace_path(self.workspace, rel)
            if not is_within_workspace(self.workspace, abs_path):
                continue
            if not os.path.exists(abs_path):
                rel_norm = os.path.relpath(abs_path, self.workspace)
                manifest["absent_files"].append(rel_norm)
                continue
            rel_norm = os.path.relpath(abs_path, self.workspace)
            backup_path = os.path.join(target_dir, rel_norm)
            os.makedirs(os.path.dirname(backup_path) or ".", exist_ok=True)
            with open(abs_path, "r", encoding="utf-8") as src:
                content = src.read()
            with open(backup_path, "w", encoding="utf-8") as dst:
                dst.write(content)
            manifest["files"].append(rel_norm)

        manifest_path = os.path.join(target_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "dir": target_dir,
            "file_count": len(manifest["files"]),
            "absent_file_count": len(manifest.get("absent_files", [])),
        }

    def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        target_dir = os.path.join(self.checkpoints_dir, checkpoint_id)
        manifest_path = os.path.join(target_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return {"ok": False, "reason": "checkpoint manifest not found", "checkpoint_id": checkpoint_id}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            return {"ok": False, "reason": f"manifest read error: {exc}", "checkpoint_id": checkpoint_id}

        restored = []
        errors = []
        for rel in manifest.get("files", []):
            backup_path = os.path.join(target_dir, rel)
            target_abs = normalize_workspace_path(self.workspace, rel)
            if not is_within_workspace(self.workspace, target_abs):
                errors.append({"file": rel, "reason": "outside workspace"})
                continue
            try:
                with open(backup_path, "r", encoding="utf-8") as src:
                    content = src.read()
                os.makedirs(os.path.dirname(target_abs) or ".", exist_ok=True)
                with open(target_abs, "w", encoding="utf-8") as dst:
                    dst.write(content)
                restored.append(rel)
            except Exception as exc:
                errors.append({"file": rel, "reason": str(exc)})
        removed_created = []
        for rel in manifest.get("absent_files", []):
            target_abs = normalize_workspace_path(self.workspace, rel)
            if not is_within_workspace(self.workspace, target_abs):
                errors.append({"file": rel, "reason": "outside workspace"})
                continue
            try:
                if os.path.exists(target_abs):
                    os.remove(target_abs)
                    removed_created.append(rel)
            except Exception as exc:
                errors.append({"file": rel, "reason": str(exc)})
        return {
            "ok": len(errors) == 0,
            "checkpoint_id": checkpoint_id,
            "restored": restored,
            "removed_created": removed_created,
            "errors": errors,
        }

    def _replace_snippet(self, original: str, op: PatchOperation) -> Tuple[Optional[str], str]:
        if not op.find:
            return None, "missing find snippet"
        matches = original.count(op.find)
        if matches != 1:
            return None, f"find match count={matches}, expected 1"
        updated = original.replace(op.find, op.replace, 1)
        if updated == original:
            return None, "no textual change produced"
        return updated, "ok"

    def _replace_symbol(self, original: str, op: PatchOperation) -> Tuple[Optional[str], str]:
        if not op.symbol or not op.new_code:
            return None, "missing symbol/new_code"
        try:
            tree = ast.parse(original)
        except Exception as exc:
            return None, f"unable to parse python AST: {exc}"

        wanted_kind = (op.symbol_kind or "").lower().strip()
        symbol = op.symbol.strip()
        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == symbol:
                if wanted_kind in {"", "function"}:
                    target_node = node
                    break
            elif isinstance(node, ast.AsyncFunctionDef) and node.name == symbol:
                if wanted_kind in {"", "async_function", "function"}:
                    target_node = node
                    break
            elif isinstance(node, ast.ClassDef) and node.name == symbol:
                if wanted_kind in {"", "class"}:
                    target_node = node
                    break

        if target_node is None:
            return None, f"symbol not found: {symbol}"

        if not getattr(target_node, "lineno", None) or not getattr(target_node, "end_lineno", None):
            return None, "symbol lineno range unavailable"

        lines = original.splitlines(keepends=True)
        start = int(target_node.lineno) - 1
        end = int(target_node.end_lineno)
        replacement = op.new_code
        if not replacement.endswith("\n"):
            replacement += "\n"
        updated = "".join(lines[:start]) + replacement + "".join(lines[end:])
        return updated, "ok"

    def _append_snippet(self, original: str, op: PatchOperation) -> Tuple[Optional[str], str]:
        snippet = op.new_code or op.replace
        if not snippet:
            return None, "missing snippet content"
        if not snippet.endswith("\n"):
            snippet += "\n"
        anchor = op.find or ""
        if not anchor:
            return original + ("\n" if not original.endswith("\n") and original else "") + snippet, "ok"
        matches = original.count(anchor)
        if matches != 1:
            return None, f"anchor match count={matches}, expected 1"
        insertion = anchor + ("\n" if not anchor.endswith("\n") else "") + snippet
        updated = original.replace(anchor, insertion, 1)
        return updated, "ok"

    def _create_file(self, target_abs: str, op: PatchOperation) -> Tuple[Optional[str], str]:
        if os.path.exists(target_abs):
            return None, "target already exists"
        content = op.new_code or op.replace or ""
        if target_abs.lower().endswith(".py"):
            try:
                ast.parse(content or "\n")
            except Exception as exc:
                return None, f"python syntax error for new file: {exc}"
        return content, "ok"

    def _delete_file(self, target_abs: str) -> Tuple[Optional[str], str]:
        if not os.path.exists(target_abs):
            return None, "file not found"
        return "", "ok"

    def _syntax_guard(self, target_abs: str, content: str) -> Tuple[bool, str]:
        if target_abs.lower().endswith(".py"):
            try:
                ast.parse(content)
                return True, "ok"
            except Exception as exc:
                return False, f"python syntax error after patch: {exc}"
        return True, "ok"

    def preview_diff(self, path_abs: str, before: str, after: str) -> str:
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        diff = difflib.unified_diff(before_lines, after_lines, fromfile=path_abs, tofile=path_abs, n=3)
        out = "".join(diff)
        return out[:12000] + ("\n...[diff truncated]..." if len(out) > 12000 else "")

    def apply_plan(self, plan: PatchPlan, dry_run: bool = True) -> Dict[str, Any]:
        results = []
        applied_files = set()
        for op in plan.operations:
            target_abs = normalize_workspace_path(self.workspace, op.file)
            if not is_within_workspace(self.workspace, target_abs):
                results.append({"file": op.file, "op": op.op_type, "ok": False, "reason": "outside workspace"})
                continue

            original, err = self._load(target_abs)
            if err and op.op_type not in {"create_file"}:
                results.append({"file": op.file, "op": op.op_type, "ok": False, "reason": err})
                continue

            if op.op_type == "replace_snippet":
                updated, reason = self._replace_snippet(original, op)
            elif op.op_type == "replace_symbol":
                updated, reason = self._replace_symbol(original, op)
            elif op.op_type == "append_snippet":
                updated, reason = self._append_snippet(original, op)
            elif op.op_type == "create_file":
                updated, reason = self._create_file(target_abs, op)
            elif op.op_type == "delete_file":
                updated, reason = self._delete_file(target_abs)
            else:
                updated, reason = None, f"unsupported op type: {op.op_type}"

            if updated is None:
                results.append({"file": op.file, "op": op.op_type, "ok": False, "reason": reason})
                continue

            if op.op_type not in {"delete_file"}:
                ok, syntax_reason = self._syntax_guard(target_abs, updated)
                if not ok:
                    results.append({"file": op.file, "op": op.op_type, "ok": False, "reason": syntax_reason})
                    continue

            if op.op_type == "create_file":
                diff_preview = self.preview_diff(target_abs, "", updated)
            elif op.op_type == "delete_file":
                diff_preview = self.preview_diff(target_abs, original, "")
            else:
                diff_preview = self.preview_diff(target_abs, original, updated)
            if not dry_run:
                if op.op_type != "create_file":
                    self._record_backup(target_abs, original)
                if op.op_type == "delete_file":
                    try:
                        os.remove(target_abs)
                    except Exception as exc:
                        results.append({"file": op.file, "op": op.op_type, "ok": False, "reason": f"delete error: {exc}"})
                        continue
                else:
                    os.makedirs(os.path.dirname(target_abs) or ".", exist_ok=True)
                    save_err = self._save(target_abs, updated)
                    if save_err:
                        results.append({"file": op.file, "op": op.op_type, "ok": False, "reason": save_err})
                        continue
                applied_files.add(op.file)

            results.append(
                {
                    "file": op.file,
                    "op": op.op_type,
                    "ok": True,
                    "reason": "applied" if not dry_run else "dry-run",
                    "diff": diff_preview,
                }
            )

        failed = [r for r in results if not r["ok"]]
        return {
            "ok": len(failed) == 0,
            "dry_run": dry_run,
            "results": results,
            "applied_files": sorted(applied_files),
            "failed_count": len(failed),
        }
