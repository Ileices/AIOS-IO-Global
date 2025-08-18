"""Aggregate node digest logs into a single cluster digest with rotation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable


class DigestAggregator:
    """Merge node digests into a single cluster level log.

    The aggregator reads JSON-lines from per-node digest files and appends
    them to a cluster-wide log file. The cluster log is rotated when it
    exceeds ``max_bytes`` or becomes older than ``max_age`` seconds.
    """

    def __init__(
        self,
        cluster_log: str = "cluster_digest.log",
        max_bytes: int = 1024 * 1024,
        max_age: int = 24 * 3600,
    ) -> None:
        self.cluster_log = Path(cluster_log)
        self.max_bytes = max_bytes
        self.max_age = max_age

    # internal helper to rotate cluster log
    def _rotate_if_needed(self) -> None:
        if not self.cluster_log.exists():
            return
        stat = self.cluster_log.stat()
        too_big = stat.st_size >= self.max_bytes
        too_old = (time.time() - stat.st_mtime) >= self.max_age
        if too_big or too_old:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            rotated = self.cluster_log.with_name(
                f"{self.cluster_log.name}.{timestamp}"
            )
            self.cluster_log.rename(rotated)

    def merge(self, node_digests: Iterable[str | Path]) -> None:
        """Merge the given ``node_digests`` into the cluster log.

        Each node digest file is read entirely and then truncated. Any missing
        files are silently skipped.
        """

        self._rotate_if_needed()
        with self.cluster_log.open("a", encoding="utf-8") as out:
            for path in node_digests:
                p = Path(path)
                if not p.exists():
                    continue
                lines = p.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    if line:
                        out.write(line + "\n")
                # clear the node log once processed
                p.write_text("")
