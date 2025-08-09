# AIOS-IO-Global

This repository contains an experimental prototype for the **AIOS IO** project. The goal is to create a self-evolving, peer-to-peer compute fabric inspired by biological concepts.

## Features

- Minimal representation of compute nodes and clusters
- Simplified Trifecta scheduler with red/blue/yellow phases
- Placeholder PulseNet communication layer
- Basic command-line interface
- Digest logging for executed tasks
- Toy RBY color encoder for data
- Task scheduling demo showing R/B/Y phases
- Experimental machine-learning algorithms under `aios_io.ml`

## Usage

```bash
# add a node and list cluster state
python -m aios_io.cli add-node mycluster node1 4 --gpu 1 --ram 8
python -m aios_io.cli list-nodes mycluster

# save and later load the cluster
python -m aios_io.cli save-cluster mycluster cluster.json
python -m aios_io.cli load-cluster cluster.json

# run a simple demo cycle
python -m aios_io.cli demo mycluster

# PulseNet networking demo
python -m aios_io.cli start-server 127.0.0.1 9000 &
python -m aios_io.cli register-peer local 127.0.0.1 9000
python -m aios_io.cli send local "hello"

# show digest logs for a cluster (JSON lines with timestamps)
python -m aios_io.cli show-digest mycluster

# encode a string into RBY colors
python - <<'PY'
from aios_io.seed import encode
print(encode(b"hello"))
PY
```

This implementation is intentionally lightweight and serves as a starting point for further development.

### Dataset Builder

The repository now includes `scripts/json_to_aios_dataset_wizard.py`, a full
wizard for converting large chat exports into trainable datasets. Run the script
directly and follow the prompts or pass `--noninteractive` with a saved config.

### Per-node Digests

Each `Node` writes its task history to a unique JSON-lines log file named
`digest_<node_id>.log`. Use the `show-digest` CLI command to view logs for a
cluster.
