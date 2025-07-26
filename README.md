# AIOS-IO-Global

This repository contains an experimental prototype for the **AIOS IO** project. The goal is to create a self-evolving, peer-to-peer compute fabric inspired by biological concepts.

## Features

- Minimal representation of compute nodes and clusters
- Simplified Trifecta scheduler with red/blue/yellow phases
- Placeholder PulseNet communication layer
- Basic command-line interface

## Usage

```bash
python -m aios_io.cli add-node mycluster node1 4 --gpu 1 --ram 8
python -m aios_io.cli list-nodes mycluster
```

This implementation is intentionally lightweight and serves as a starting point for further development.
