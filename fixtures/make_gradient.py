#!/usr/bin/env python3
"""Generate the deterministic red/green vision fixture used by the live gate."""

from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path


WIDTH = 64
HEIGHT = 64
OUTPUT = Path(__file__).with_name("vision-gradient.png")


def chunk(name: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk."""
    return (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", binascii.crc32(name + payload) & 0xFFFFFFFF)
    )


def main() -> None:
    rows = bytearray()
    for y in range(HEIGHT):
        rows.append(0)
        for x in range(WIDTH):
            red = round(255 * x / (WIDTH - 1))
            green = round(255 * y / (HEIGHT - 1))
            rows.extend((red, green, 128))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + chunk(b"IEND", b"")
    )
    OUTPUT.write_bytes(png)
    print(OUTPUT)


if __name__ == "__main__":
    main()
