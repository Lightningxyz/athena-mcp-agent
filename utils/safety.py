import os
import shlex
from typing import List, Tuple

from config import (
    AUTOFIX_ALLOWED_COMMAND_PREFIXES_CSV,
    AUTOFIX_BLOCKED_COMMAND_TERMS_CSV,
    AUTOFIX_REQUIRE_APPROVAL_FOR_HIGH_RISK,
)


def _csv_to_list(raw_csv: str) -> List[str]:
    return [x.strip() for x in raw_csv.split(",") if x.strip()]


ALLOWED_PREFIXES = _csv_to_list(AUTOFIX_ALLOWED_COMMAND_PREFIXES_CSV)
BLOCKED_TERMS = [x.lower() for x in _csv_to_list(AUTOFIX_BLOCKED_COMMAND_TERMS_CSV)]


def normalize_workspace_path(workspace: str, target: str) -> str:
    workspace_abs = os.path.abspath(workspace)
    candidate = target if os.path.isabs(target) else os.path.join(workspace_abs, target)
    return os.path.abspath(candidate)


def is_within_workspace(workspace: str, target_abs: str) -> bool:
    workspace_abs = os.path.abspath(workspace)
    return target_abs == workspace_abs or target_abs.startswith(workspace_abs + os.sep)


def validate_command_safety(cmd: str) -> Tuple[bool, str]:
    stripped = (cmd or "").strip()
    if not stripped:
        return False, "empty command"

    lower = stripped.lower()
    for term in BLOCKED_TERMS:
        if term and term in lower:
            return False, f"blocked term detected: {term}"

    # reject shell composition operators for now
    for op in ["&&", "||", ";", "|", ">", "<", "$(", "`"]:
        if op in stripped:
            return False, f"shell operator not allowed: {op}"

    try:
        parts = shlex.split(stripped)
    except Exception as exc:
        return False, f"unable to parse command: {exc}"

    if not parts:
        return False, "empty command after parse"

    normalized = " ".join(parts)
    for prefix in ALLOWED_PREFIXES:
        if normalized.startswith(prefix):
            return True, "ok"

    return False, "command prefix not in allowlist"


def requires_high_risk_approval(risk: str, user_override: bool) -> Tuple[bool, str]:
    if not AUTOFIX_REQUIRE_APPROVAL_FOR_HIGH_RISK:
        return False, "high-risk approval disabled in config"
    if (risk or "").lower() != "high":
        return False, "risk is not high"
    if user_override:
        return False, "user explicitly approved high risk"
    return True, "high-risk patch requires --approve-high-risk"
