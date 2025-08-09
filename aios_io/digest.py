"""Structured JSON-lines digest logger for task results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class Digest:
    """Logs executed tasks to a JSON-lines file."""

    def __init__(self, path: str = "digest.log") -> None:
        self.path = Path(path)

    def log(self, entry: Dict[str, object]) -> None:
        """Append a JSON serializable ``entry`` to the digest file."""

        with self.path.open("a", encoding="utf-8") as f:
            json.dump(entry, f)
            f.write("\n")

    def read(self) -> List[Dict[str, object]]:
        """Return all log entries as dictionaries."""

        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line]
