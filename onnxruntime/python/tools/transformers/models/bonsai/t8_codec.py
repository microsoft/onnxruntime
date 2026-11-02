# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

LAYOUT = "unit5x128-planes-v1"
GROUP_SIZE = 128
ROWS_PER_UNIT = 5
POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def encode_q2(packed: np.ndarray, scales: np.ndarray, *, verify: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Losslessly re-encode MatMulNBits W2, zero point 1, into the documented T8 layout."""
    if packed.dtype != np.uint8 or packed.ndim != 3 or packed.shape[2] != 32:
        raise ValueError("Expected uint8 Q2 weights [N, K/128, 32]")
    rows, groups, _ = packed.shape
    if rows <= 0 or groups <= 0 or scales.shape not in ((rows, groups), (rows * groups,)):
        raise ValueError("Invalid Q2 dimensions or scales shape")
    scales = scales.reshape(rows, groups)
    s16 = scales.astype(np.float16)
    if not np.all(np.isfinite(scales)) or not np.array_equal(s16.astype(scales.dtype), scales):
        raise ValueError("Scales must be finite and exactly representable as float16")
    units, width = (rows + 4) // 5, groups * GROUP_SIZE
    planes = np.empty((units, 8, width // 32), dtype="<u4")
    t8_scales = np.zeros((units, groups, 5), dtype=np.float16)
    for start in range(0, units, 128):
        end = min(start + 128, units)
        row_start, row_end = start * 5, min(end * 5, rows)
        source = packed[row_start:row_end].reshape(row_end - row_start, -1)
        codes = ((source[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1, width)
        if np.any(codes == 3):
            raise ValueError(f"Non-ternary Q2 code 3 in rows {row_start}..{row_end - 1}")
        trits = np.zeros(((end - start) * 5, width), dtype=np.uint8)
        trits[: row_end - row_start] = (codes + 2) % 3
        values = (trits.reshape(end - start, 5, width) * POW3[None, :, None]).sum(axis=1).astype(np.uint8)
        bits = (values.reshape(end - start, -1, 32, 1) >> np.arange(8, dtype=np.uint8)) & 1
        bits = bits.transpose(0, 3, 1, 2)
        chunk = np.ascontiguousarray(np.packbits(bits, axis=-1, bitorder="little")).view("<u4")
        planes[start:end] = chunk.reshape(end - start, 8, width // 32)
        padded_scales = np.zeros(((end - start) * 5, groups), dtype=np.float16)
        padded_scales[: row_end - row_start] = s16[row_start:row_end]
        t8_scales[start:end] = padded_scales.reshape(-1, 5, groups).transpose(0, 2, 1)
        if verify:
            restored = decode_q2(planes[start:end], row_end - row_start)
            if not np.array_equal(restored, packed[row_start:row_end]):
                raise ValueError(f"T8 round-trip failed in rows {row_start}..{row_end - 1}")
    return planes, t8_scales


def decode_q2(planes: np.ndarray, rows: int) -> np.ndarray:
    """Expand a T8 matrix back to the original affine Q2 codes (not floating-point weights)."""
    if planes.dtype != np.uint32 or planes.ndim != 3 or planes.shape[1] != 8:
        raise ValueError("Expected uint32 T8 bitplanes [ceil(N/5), 8, K/32]")
    if rows <= 0 or (rows + 4) // 5 != planes.shape[0] or planes.shape[2] % 4:
        raise ValueError("Invalid T8 dimensions")
    units, _, sets = planes.shape
    raw = np.ascontiguousarray(planes, dtype="<u4").view(np.uint8).reshape(units, 8, sets, 4)
    bits = np.unpackbits(raw, axis=-1, bitorder="little").transpose(0, 2, 3, 1)
    values = (bits << np.arange(8, dtype=np.uint8)).sum(axis=-1).astype(np.uint8).reshape(units, sets * 32)
    if np.any(values > 242):
        raise ValueError("Invalid T8 byte (243..255)")
    trits = ((values[:, None, :] // POW3[None, :, None]) % 3).reshape(units * 5, -1)[:rows]
    codes = ((trits + 1) % 3).astype(np.uint8).reshape(rows, -1, 4)
    packed = (codes << np.array([0, 2, 4, 6], dtype=np.uint8)).sum(axis=-1).astype(np.uint8)
    return packed.reshape(rows, sets // 4, 32)
