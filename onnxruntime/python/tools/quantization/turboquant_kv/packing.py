# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Bit-packing utilities for TurboQuant.

Two layouts:
  - 4-bit: pairs of values packed into one uint8. ceil(D/2) bytes.
  - 3-bit: groups of 8 values packed into 3 uint8 bytes (24 bits exactly).

The 3-bit layout has no native equivalent in ONNX standard ops or in WGSL —
both runtime EPs need a custom unpack helper. This file is the canonical
reference for the bit-ordering convention.

Convention for 3-bit packing of 8 values (v0..v7), each in [0, 7]:
  Treat the 8 values as a 24-bit little-endian integer:
    bits  [0:3]  = v0
    bits  [3:6]  = v1
    bits  [6:9]  = v2
    bits  [9:12] = v3
    bits [12:15] = v4
    bits [15:18] = v5
    bits [18:21] = v6
    bits [21:24] = v7
  Then split into 3 bytes, byte k = bits [8k : 8k+8] of the 24-bit integer.

This is the same convention vLLM uses (see triton_turboquant_store.py:304-315).
"""

from __future__ import annotations

import math

import numpy as np


def packed_size_bytes(d: int, bits: int) -> int:
    """Bytes required to store D values quantized to `bits` bits each."""
    return math.ceil(d * bits / 8)


# ----------------------------------------------------------------------------
# 4-bit: pairs into uint8 (lower nibble = even index, upper nibble = odd index)
# ----------------------------------------------------------------------------


def pack_4bit(values: np.ndarray) -> np.ndarray:
    """Pack values in [0, 15] into uint8, two-per-byte.

    Args:
        values: Shape (..., D). Last dim D must be even. Dtype convertible to uint8.

    Returns:
        Packed: Shape (..., D//2), uint8.
        Lower nibble of byte i = values[..., 2*i].
        Upper nibble of byte i = values[..., 2*i + 1].
    """
    v = values.astype(np.uint8)
    if v.shape[-1] % 2 != 0:
        raise ValueError(f"4-bit packing requires even last dim, got {v.shape[-1]}")
    if v.max(initial=0) > 15:
        raise ValueError("4-bit values must be in [0, 15]")
    lo = v[..., 0::2]
    hi = v[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_4bit(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack 4-bit packed bytes into D values per row.

    Args:
        packed: Shape (..., ceil(D/2)), uint8.
        d: Logical number of values per row.

    Returns:
        Shape (..., D), uint8 values in [0, 15].
    """
    if d % 2 != 0:
        raise ValueError(f"4-bit unpacking requires even D, got {d}")
    if packed.shape[-1] != d // 2:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} doesn't match D//2={d // 2}"
        )
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = np.empty(packed.shape[:-1] + (d,), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


# ----------------------------------------------------------------------------
# 3-bit: 8 values into 3 bytes (24 bits exactly).
# ----------------------------------------------------------------------------


def pack_3bit(values: np.ndarray) -> np.ndarray:
    """Pack values in [0, 7] into uint8 bytes, 8 values per 3 bytes.

    Args:
        values: Shape (..., D). Last dim D must be a multiple of 8.

    Returns:
        Packed: Shape (..., D * 3 // 8), uint8.

    Layout: each group of 8 consecutive values becomes a 24-bit little-endian
    integer (v0 in bits [0:3], v7 in bits [21:24]), split into 3 bytes.
    """
    v = values.astype(np.uint8)
    if v.shape[-1] % 8 != 0:
        raise ValueError(
            f"3-bit packing requires last dim divisible by 8, got {v.shape[-1]}"
        )
    if v.max(initial=0) > 7:
        raise ValueError("3-bit values must be in [0, 7]")

    n_groups = v.shape[-1] // 8
    # Reshape so the 8-element groups are along a new axis.
    g = v.reshape(*v.shape[:-1], n_groups, 8).astype(np.uint32)
    # Compose 24-bit words.
    words = (
        (g[..., 0])
        | (g[..., 1] << 3)
        | (g[..., 2] << 6)
        | (g[..., 3] << 9)
        | (g[..., 4] << 12)
        | (g[..., 5] << 15)
        | (g[..., 6] << 18)
        | (g[..., 7] << 21)
    )  # shape (..., n_groups), uint32, only low 24 bits used
    # Split into 3 bytes per word.
    b0 = (words & 0xFF).astype(np.uint8)
    b1 = ((words >> 8) & 0xFF).astype(np.uint8)
    b2 = ((words >> 16) & 0xFF).astype(np.uint8)
    out = np.stack([b0, b1, b2], axis=-1)  # (..., n_groups, 3)
    return out.reshape(*v.shape[:-1], n_groups * 3)


def unpack_3bit(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack 3-bit packed bytes back into D values per row.

    Args:
        packed: Shape (..., D * 3 // 8), uint8.
        d: Logical number of values per row. Must be a multiple of 8.

    Returns:
        Shape (..., D), uint8 values in [0, 7].
    """
    if d % 8 != 0:
        raise ValueError(f"3-bit unpacking requires D divisible by 8, got {d}")
    expected_bytes = d * 3 // 8
    if packed.shape[-1] != expected_bytes:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} doesn't match expected {expected_bytes}"
        )
    n_groups = d // 8
    p = packed.reshape(*packed.shape[:-1], n_groups, 3).astype(np.uint32)
    words = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    out = np.empty(packed.shape[:-1] + (n_groups, 8), dtype=np.uint8)
    for i in range(8):
        out[..., i] = ((words >> (i * 3)) & 0x7).astype(np.uint8)
    return out.reshape(*packed.shape[:-1], d)


# ----------------------------------------------------------------------------
# Generic dispatcher.
# ----------------------------------------------------------------------------


def pack(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack values to `bits` bits per element. Supports 3 and 4."""
    if bits == 3:
        return pack_3bit(values)
    if bits == 4:
        return pack_4bit(values)
    raise NotImplementedError(f"Only 3-bit and 4-bit packing supported, got {bits}")


def unpack(packed: np.ndarray, d: int, bits: int) -> np.ndarray:
    """Unpack `bits`-bit packed values back to a (..., D) array."""
    if bits == 3:
        return unpack_3bit(packed, d)
    if bits == 4:
        return unpack_4bit(packed, d)
    raise NotImplementedError(f"Only 3-bit and 4-bit unpacking supported, got {bits}")
