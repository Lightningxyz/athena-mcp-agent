import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from config import SESSION_MEMORY_FILE, SESSION_MEMORY_MAX_ENTRIES


@dataclass
class MemoryEntry:
    timestamp: float
    query: str
    summary: str
    status: str
    files_touched: List[str]
    confidence: str


class SessionMemory:
    def __init__(self, workspace: str, memory_file: str = SESSION_MEMORY_FILE):
        self.workspace = os.path.abspath(workspace)
        self.path = memory_file if os.path.isabs(memory_file) else os.path.join(self.workspace, memory_file)
        self._data: Dict[str, Any] = {"entries": []}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._data = {"entries": []}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict) and isinstance(raw.get("entries"), list):
                    self._data = raw
                else:
                    self._data = {"entries": []}
        except Exception:
            self._data = {"entries": []}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def add_entry(
        self,
        query: str,
        summary: str,
        status: str,
        files_touched: List[str],
        confidence: str = "unknown",
    ) -> None:
        entry = MemoryEntry(
            timestamp=time.time(),
            query=query,
            summary=summary,
            status=status,
            files_touched=files_touched,
            confidence=confidence,
        )
        entries = self._data.get("entries", [])
        entries.append(asdict(entry))
        if len(entries) > SESSION_MEMORY_MAX_ENTRIES:
            entries = entries[-SESSION_MEMORY_MAX_ENTRIES:]
        self._data["entries"] = entries
        self._save()

    def recent_entries(self, n: int = 5) -> List[Dict[str, Any]]:
        entries = self._data.get("entries", [])
        return entries[-max(0, n):]

    def summarize_recent(self, n: int = 6) -> str:
        items = self.recent_entries(n)
        if not items:
            return "No prior session memory."
        lines = []
        for idx, item in enumerate(items, 1):
            query = str(item.get("query", "")).strip()
            summary = str(item.get("summary", "")).strip()
            status = str(item.get("status", "unknown")).strip()
            files = item.get("files_touched", [])
            file_hint = ", ".join(files[:3]) if isinstance(files, list) and files else "none"
            lines.append(f"{idx}. [{status}] query={query} | summary={summary} | files={file_hint}")
        return "\n".join(lines)
