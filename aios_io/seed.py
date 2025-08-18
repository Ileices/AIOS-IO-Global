"""Configurable RBY spectral encoder/decoder."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

Color = Tuple[int, int, int]  # (R, G, B)


def _permutations(key: int) -> Tuple[List[int], List[int], List[int]]:
    """Return deterministic byte-to-color permutations based on ``key``."""
    rng = random.Random(key)
    base = list(range(256))
    r_perm = base.copy()
    g_perm = base.copy()
    b_perm = base.copy()
    rng.shuffle(r_perm)
    rng.shuffle(g_perm)
    rng.shuffle(b_perm)
    return r_perm, g_perm, b_perm


def encode(data: bytes, key: int = 0) -> List[Color]:
    """Encode bytes into RGB colors using key-based permutations."""
    r_perm, g_perm, b_perm = _permutations(key)
    return [(r_perm[b], g_perm[b], b_perm[b]) for b in data]


def decode(colors: List[Color], key: int = 0) -> bytes:
    """Decode RGB colors back into bytes, validating against ``key``."""
    r_perm, g_perm, b_perm = _permutations(key)
    inv_r: Dict[int, int] = {v: i for i, v in enumerate(r_perm)}
    inv_g: Dict[int, int] = {v: i for i, v in enumerate(g_perm)}
    inv_b: Dict[int, int] = {v: i for i, v in enumerate(b_perm)}
    data = bytearray()
    for r, g, b in colors:
        byte = inv_r.get(r)
        if byte is None or inv_g.get(g) != byte or inv_b.get(b) != byte:
            raise ValueError("color triple does not match key")
        data.append(byte)
    return bytes(data)
