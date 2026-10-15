# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared helpers for the onnx-ir / onnxscript-rewriter based fusions."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

FLOAT_OR_FLOAT16 = frozenset((ir.DataType.FLOAT, ir.DataType.FLOAT16))
FLOAT_OR_FLOAT16_OR_BFLOAT16 = FLOAT_OR_FLOAT16 | {ir.DataType.BFLOAT16}


def constant_array(value: ir.Value) -> np.ndarray | None:
    """Return the constant tensor backing ``value`` as a numpy array, or None.

    Works for both graph initializers and ``Constant`` node outputs (onnx-ir
    exposes both through ``Value.const_value``).
    """
    const = value.const_value
    if const is None:
        return None
    return const.numpy()


def scalar_constant(value: ir.Value) -> float | None:
    """Return ``value`` as a Python float if it is a scalar/1-element constant."""
    array = constant_array(value)
    if array is None:
        return None
    flat = array.reshape(-1)
    return float(flat[0]) if flat.size == 1 else None


def is_constant_with_rank(value: ir.Value, rank: int) -> bool:
    """True if ``value`` is a constant tensor with the given number of dims."""
    array = constant_array(value)
    return array is not None and array.ndim == rank


def static_shape(value: ir.Value) -> tuple[int, ...] | None:
    """Return the fully static shape of ``value``, or ``None`` if it is unavailable."""
    shape = value.shape
    if shape is None or not shape.is_static():
        return None
    return shape.numpy()
