"""Standalone packed-format numerical oracles, not a production QMoE API.

Generic uint8 codecs define no QMoE execution or ABI support. Logical rows
restart on byte boundaries; only the profile validator requires canonical pads.
"""

import sys
from dataclasses import dataclass

import numpy as np

SIZE_LIMIT = min(sys.maxsize, int(np.iinfo(np.int64).max))


def checked_integer(value, name, minimum=0, maximum=SIZE_LIMIT):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not bool")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if value > maximum:
        raise OverflowError(f"{name} exceeds {maximum}")
    return value


def checked_shape(shape, itemsize=1):
    """Check dimensions and addressable strides even for zero-sized tensors."""
    itemsize = checked_integer(itemsize, "itemsize", 1)
    result = tuple(checked_integer(dim, "dimension") for dim in shape)
    extent = itemsize
    for dim in result:
        factor = max(1, dim)
        if extent > SIZE_LIMIT // factor:
            raise OverflowError("shape exceeds addressable size")
        extent *= factor
    return result


def checked_width(bits):
    bits = checked_integer(bits, "bits", 1)
    if bits > 8:
        raise ValueError("bits must be in 1..8")
    return bits


def row_bytes(count, bits):
    """Compute ceil(count * bits / 8) without an overflowing product."""
    bits = checked_width(bits)
    count = checked_integer(count, "count")
    whole, remainder = divmod(count, 8)
    tail = (remainder * bits + 7) // 8
    if whole > (SIZE_LIMIT - tail) // bits:
        raise OverflowError("packed row exceeds addressable size")
    return whole * bits + tail


def _uint8_rows(values):
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.uint8):
        raise TypeError("input must be a numpy uint8 array")
    if values.ndim == 0:
        raise ValueError("input must have rank >= 1")
    checked_shape(values.shape)


def pack(codes, bits):
    """Pack unsigned codes along the last axis, LSB first, with zero padding."""
    bits = checked_width(bits)
    _uint8_rows(codes)
    if np.any(codes > (1 << bits) - 1):
        raise ValueError("code is outside the unsigned width")
    count = codes.shape[-1]
    shape = checked_shape((*codes.shape[:-1], row_bytes(count, bits)))
    packed = np.zeros(shape, dtype=np.uint8)
    if codes.size == 0:
        return packed
    for index in range(count):
        byte, shift = divmod(index * bits, 8)
        word = codes[..., index].astype(np.uint16) << shift
        packed[..., byte] |= (word & 255).astype(np.uint8)
        if shift + bits > 8:
            packed[..., byte + 1] |= (word >> 8).astype(np.uint8)
    return packed


def unpack(packed, bits, count):
    """Unpack exactly count codes per row, ignoring unused high padding bits."""
    bits = checked_width(bits)
    count = checked_integer(count, "count")
    _uint8_rows(packed)
    shape = checked_shape((*packed.shape[:-1], count))
    if packed.shape[-1] != row_bytes(count, bits):
        raise ValueError("packed last axis must have the exact byte count")
    codes = np.empty(shape, dtype=np.uint8)
    if codes.size == 0:
        return codes
    for index in range(count):
        byte, shift = divmod(index * bits, 8)
        word = packed[..., byte].astype(np.uint16)
        if shift + bits > 8:
            word = word | (packed[..., byte + 1].astype(np.uint16) << 8)
        codes[..., index] = ((word >> shift) & ((1 << bits) - 1)).astype(np.uint8)
    return codes


def validate_canonical_padding(packed, bits, count):
    """Reject nonzero unused high bits separately from generic unpack semantics."""
    bits = checked_width(bits)
    count = checked_integer(count, "count")
    _uint8_rows(packed)
    if packed.shape[-1] != row_bytes(count, bits):
        raise ValueError("packed last axis must have the exact byte count")
    used = ((count % 8) * bits) % 8
    if used and np.any(packed[..., -1] & (255 ^ ((1 << used) - 1))):
        raise ValueError("nonzero canonical padding")


def checked_block_size(block_size):
    block_size = checked_integer(block_size, "block_size", 1)
    if block_size not in (32, 64, 128):
        raise ValueError("baseline block_size must be 32, 64, or 128")
    return block_size


def block_count(count, block_size):
    count = checked_integer(count, "count")
    block_size = checked_block_size(block_size)
    whole, tail = divmod(count, block_size)
    return whole + bool(tail)


def _scale_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float16)) and dtype.name != "bfloat16":
        raise TypeError("scales must use float32, float16, or available bfloat16")
    return dtype


def validate_int3(packed, scales, count, block_size, zeros=None):
    """Validate the proposed INT3 numerical profile, including reference-only tails.

    This is deliberately stricter than generic unpack and does not validate any
    current QMoE operator schema. Explicit zeros encode B block codes, not K.
    """
    count = checked_integer(count, "count")
    blocks = block_count(count, block_size)
    _uint8_rows(packed)
    if packed.ndim != 3:
        raise ValueError("INT3 weights must have rank 3 [E,N,row_bytes]")
    validate_canonical_padding(packed, 3, count)
    checked_shape((*packed.shape[:-1], count), 4)
    if not isinstance(scales, np.ndarray):
        raise TypeError("scales must be a numpy array")
    _scale_dtype(scales.dtype)
    if scales.shape != (*packed.shape[:-1], blocks):
        raise ValueError("scales must match [E,N,ceil(K/block_size)]")
    checked_shape(scales.shape, scales.dtype.itemsize)
    if not np.all(np.isfinite(scales)) or np.any(scales < 0):
        raise ValueError("scales must be finite and nonnegative")
    if zeros is not None:
        _uint8_rows(zeros)
        if zeros.shape != (*packed.shape[:-1], row_bytes(blocks, 3)):
            raise ValueError("zeros must match [E,N,ceil(3*B/8)]")
        validate_canonical_padding(zeros, 3, blocks)
        if np.any(unpack(zeros, 3, blocks) != 4):
            raise ValueError("explicit INT3 zero point codes must all equal 4")


def decode_int3(packed, scales, count, block_size, zeros=None):
    """Decode to float32 with cast-before-subtract and logical block gathers."""
    validate_int3(packed, scales, count, block_size, zeros)
    codes = unpack(packed, 3, count).astype(np.float32)
    if codes.size == 0:
        return codes
    indices = np.arange(count, dtype=np.int64) // block_size
    expanded_scales = np.take(scales.astype(np.float32), indices, axis=-1)
    zero = np.float32(4)
    if zeros is not None:
        blocks = block_count(count, block_size)
        zero = np.take(unpack(zeros, 3, blocks).astype(np.float32), indices, axis=-1)
    return (codes - zero) * expanded_scales


@dataclass(frozen=True)
class RtnResult:
    packed: np.ndarray
    scales: np.ndarray
    underflow_blocks: np.ndarray
    max_abs_error: np.ndarray


def quantize_int3_rtn(weights, block_size, scale_dtype=np.float32):
    """Deterministic fixture RTN, not a production quantizer.

    Reductions/division use float64. Scales are rounded to their final storage
    dtype before nearest-even code generation and clipping to [-4,3]. Zero or
    underflow groups emit code 4. Per-block error and underflow flags are returned
    without silently promoting the requested storage dtype.
    """
    block_size = checked_block_size(block_size)
    dtype = _scale_dtype(scale_dtype)
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a numpy floating-point array")
    if weights.ndim != 3:
        raise ValueError("weights must have rank 3 [E,N,K]")
    if weights.dtype.kind != "f" and weights.dtype.name != "bfloat16":
        raise TypeError("weights must be floating-point")
    checked_shape(weights.shape, max(8, weights.dtype.itemsize))
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    count = weights.shape[-1]
    blocks = block_count(count, block_size)
    shape = checked_shape((*weights.shape[:-1], blocks), 8)
    scales = np.zeros(shape, dtype=dtype)
    underflow = np.zeros(shape, dtype=bool)
    errors = np.zeros(shape, dtype=np.float64)
    codes = np.full(weights.shape, 4, dtype=np.uint8)
    if weights.size == 0:
        return RtnResult(pack(codes, 3), scales, underflow, errors)
    for block in range(blocks):
        start = block * block_size
        stop = min(start + block_size, count)
        group = weights[..., start:stop].astype(np.float64)
        raw_scale = np.maximum(np.max(np.maximum(-group, 0), axis=-1) / 4, np.max(np.maximum(group, 0), axis=-1) / 3)
        with np.errstate(over="ignore", under="ignore"):
            stored = raw_scale.astype(dtype)
        if not np.all(np.isfinite(stored)):
            raise ValueError("scale overflows the requested storage dtype")
        scales[..., block] = stored
        rounded = stored.astype(np.float64)
        underflow[..., block] = (raw_scale > 0) & (rounded == 0)
        normalized = np.divide(group, rounded[..., None], out=np.zeros_like(group), where=rounded[..., None] != 0)
        signed = np.clip(np.rint(normalized), -4, 3)
        codes[..., start:stop] = (signed + 4).astype(np.uint8)
        errors[..., block] = np.max(np.abs(group - signed * rounded[..., None]), axis=-1)
    return RtnResult(pack(codes, 3), scales, underflow, errors)
