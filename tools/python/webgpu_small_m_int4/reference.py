# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Independent byte-level INT4 decoding and float64 reference calculations."""

import numpy as np


def pack_nibbles(values):
    values = np.asarray(values)
    if values.dtype != np.uint8 or values.shape[-1] % 2 or np.any(values > 15):
        raise ValueError("Expected uint8 nibbles and an even final dimension.")
    return values[..., ::2] | (values[..., 1::2] << 4)


def pack_zero_points(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype != np.uint8 or np.any(values > 15):
        raise ValueError("Expected uint8 zero points with shape [N, blocks].")
    result = np.zeros((values.shape[0], (values.shape[1] + 1) // 2), np.uint8)
    result[:, : values[:, ::2].shape[1]] = values[:, ::2]
    result[:, : values[:, 1::2].shape[1]] |= values[:, 1::2] << 4
    return result


def projection(a, packed, scales, k, block_size, zero_points=None, bias=None):
    """Return FP64, FP64-with-FP16-weights, and absolute-product-sum references."""
    a = np.asarray(a)
    packed = np.asarray(packed)
    if a.shape[-1] != k or packed.dtype != np.uint8:
        raise ValueError("Incorrect activation K dimension or packed-weight dtype.")
    n = packed.shape[0]
    single_scale = block_size == k * n
    per_column = k if single_scale else block_size
    blocks = (k + per_column - 1) // per_column
    padded_k = blocks * per_column
    if per_column % 2 or packed.size != n * padded_k // 2:
        raise ValueError("Incorrect packed-weight size.")
    raw = packed.reshape(n, padded_k // 2)
    scale = np.asarray(scales, np.float64)
    if scale.size != (1 if single_scale else n * blocks):
        raise ValueError("Incorrect scale count.")
    scale = scale.reshape(1, 1) if single_scale else scale.reshape(n, blocks)
    zero = np.full((n, blocks), 8.0, np.float64)
    if zero_points is not None:
        zp = np.asarray(zero_points).reshape(n, (blocks + 1) // 2)
        zero[:, ::2] = zp[:, : zero[:, ::2].shape[1]] & 15
        zero[:, 1::2] = zp[:, : zero[:, 1::2].shape[1]] >> 4
        if single_scale:
            zero.fill(zero[0, 0])
    a64 = a.astype(np.float64).reshape(-1, k)
    full = np.empty((a64.shape[0], n), np.float64)
    rounded = np.empty_like(full)
    absolute = np.empty_like(full)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        q = np.empty((stop - start, padded_k), np.float64)
        q[:, ::2] = raw[start:stop] & 15
        q[:, 1::2] = raw[start:stop] >> 4
        weights = q.reshape(stop - start, blocks, per_column)
        weights -= zero[start:stop, :, None]
        weights *= scale[0 if single_scale else start : 1 if single_scale else stop, :, None]
        weights = weights.reshape(stop - start, padded_k)[:, :k]
        half_weights = weights.astype(np.float16).astype(np.float64)
        full[:, start:stop] = a64 @ weights.T
        rounded[:, start:stop] = a64 @ half_weights.T
        absolute[:, start:stop] = np.abs(a64) @ np.abs(half_weights).T
    if bias is not None:
        bias64 = np.asarray(bias, np.float64).reshape(1, n)
        full += bias64
        rounded += bias64
    shape = (*a.shape[:-1], n)
    return full.reshape(shape), rounded.reshape(shape), absolute.reshape(shape)


def errors(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"Output/reference shape mismatch: {actual.shape}, {expected.shape}")
    nonfinite = int(np.count_nonzero(~np.isfinite(actual)))
    expected_nonfinite = int(np.count_nonzero(~np.isfinite(expected)))
    if nonfinite or expected_nonfinite:
        return {
            "finite": False,
            "nonfinite": nonfinite,
            "expected_nonfinite": expected_nonfinite,
            "max_abs": None,
            "relative_l2": None,
        }
    delta = actual - expected
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return {
        "finite": True,
        "nonfinite": 0,
        "max_abs": float(np.max(np.abs(delta), initial=0)),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "rms_abs": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
    }


def assert_fp32_accuracy(actual, rounded_reference, absolute_products, k):
    """Check an association-aware forward-error bound plus final FP16 rounding."""
    actual = np.asarray(actual, np.float64)
    target = rounded_reference.astype(np.float16).astype(np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise AssertionError("The finite-output correctness fixtures produced nonfinite outputs.")
    eps = np.finfo(np.float32).eps
    operations = k / 8 + 16
    gamma = operations * eps / (1 - operations * eps)
    rounding = np.abs(target - rounded_reference)
    bound = rounding + 2 * gamma * absolute_products + np.finfo(np.float16).smallest_subnormal
    delta = np.abs(actual - rounded_reference)
    if np.any(delta > bound):
        index = np.unravel_index(np.argmax(delta - bound), delta.shape)
        raise AssertionError(
            f"FP32/FP16 forward-error bound exceeded at {index}: "
            f"actual={actual[index]}, reference={rounded_reference[index]}, bound={bound[index]}"
        )
