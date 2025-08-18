import json
from pathlib import Path

from aios_io.digest_aggregator import DigestAggregator


def test_digest_aggregator_merges_and_rotates(tmp_path):
    node1 = tmp_path / "n1.log"
    node2 = tmp_path / "n2.log"
    node1.write_text(json.dumps({"node": "n1", "task": "a"}) + "\n")
    node2.write_text(json.dumps({"node": "n2", "task": "b"}) + "\n")

    cluster_log = tmp_path / "cluster.log"
    agg = DigestAggregator(str(cluster_log), max_bytes=10, max_age=3600)
    agg.merge([node1, node2])

    # node logs should be cleared after merge
    assert node1.read_text() == ""
    assert node2.read_text() == ""

    lines = cluster_log.read_text().splitlines()
    assert len(lines) == 2

    # write another entry to trigger rotation by size
    node1.write_text(json.dumps({"node": "n1", "task": "c"}) + "\n")
    agg.merge([node1])
    rotated = list(tmp_path.glob("cluster.log.*"))
    assert rotated, "cluster log should rotate when exceeding size"
