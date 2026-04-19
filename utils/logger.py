from typing import List, Dict, Any
import json
import os
import time
from utils.spinner import Spinner
from config import TRACE_JSONL_FILE

class TraceLogger:
    def __init__(self, spinner: Spinner = None):
        self.trace: List[Dict[str, Any]] = []
        self.spinner = spinner
        
    def log(self, event_type: str, details: Dict[str, Any]):
        self.trace.append({"event": event_type, "details": details})
        if self.spinner:
            action = details.get("action", "")
            action_str = f" - {action.replace('_', ' ').title()}" if action else ""
            self.spinner.update(f"Running: {event_type.replace('_', ' ').title()}{action_str}...")
        
    def get_summary(self) -> str:
        return json.dumps(self.trace, indent=2)

    def export_jsonl(self, path: str = TRACE_JSONL_FILE) -> str:
        target = os.path.abspath(path)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            for item in self.trace:
                row = {
                    "ts": time.time(),
                    "event": item.get("event"),
                    "details": item.get("details", {}),
                }
                f.write(json.dumps(row) + "\n")
        return target

def print_trace(logger: TraceLogger):
    print("\n--- Trace ---")
    print(logger.get_summary())
