"""Simple digest logger for task results."""
from __future__ import annotations

from pathlib import Path
from typing import List


class Digest:
    """Logs executed tasks to a file."""

    def __init__(self, path: str = "digest.log") -> None:
        self.path = Path(path)

    def log(self, entry: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def read(self) -> List[str]:
        if not self.path.exists():
            return []
        return self.path.read_text(encoding="utf-8").splitlines()
