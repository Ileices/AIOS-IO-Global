"""Simple RBY spectral encoder/decoder."""
from __future__ import annotations

from typing import List, Tuple


Color = Tuple[int, int, int]  # (R, G, B)


def encode(data: bytes) -> List[Color]:
    """Encode bytes into a list of RGB colors using a toy mapping."""
    colors: List[Color] = []
    for b in data:
        r = b
        g = (b * 2) % 256
        blue = (b * 3) % 256
        colors.append((r, g, blue))
    return colors


def decode(colors: List[Color]) -> bytes:
    """Decode colors back into bytes assuming the encode mapping."""
    data = bytearray()
    for r, _g, _b in colors:
        data.append(r)
    return bytes(data)
