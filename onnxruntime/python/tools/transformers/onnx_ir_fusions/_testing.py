# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared test helpers for onnx_ir_fusions unit tests."""

from __future__ import annotations

from collections import Counter

import onnx
import onnx_ir as ir


def op_counts(model: ir.Model) -> Counter:
    """Count node op_types in an ir.Model graph."""
    return Counter(node.op_type for node in model.graph)


def to_ir(model_proto: onnx.ModelProto) -> ir.Model:
    """Convert an onnx.ModelProto to an ir.Model."""
    return ir.from_proto(model_proto)
