"""Small, fixed-shape semantic graphs using only standard ONNX opset 18.

These CPU reference graphs are not QMoE models or runtime support claims. Call
validate_int3 on profile inputs first: the graph expresses decoding, not host
validation of finite scales, canonical padding, or all-4 explicit zero points.
"""

import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper
from reference import block_count, checked_integer, checked_shape, checked_width, row_bytes


class _Graph:
    def __init__(self):
        self.nodes = []
        self.initializers = []

    def constant(self, name, value):
        self.initializers.append(numpy_helper.from_array(np.asarray(value), name=name))
        return name

    def node(self, op, inputs, output, **attributes):
        self.nodes.append(helper.make_node(op, inputs, [output], **attributes))
        return output

    def model(self, inputs, output, dtype, shape):
        graph = helper.make_graph(
            self.nodes,
            "packed_reference",
            inputs,
            [helper.make_tensor_value_info(output, dtype, shape)],
            self.initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="p0_reference")
        checker.check_model(model, full_check=True)
        return model


def _packed_shape(shape, bits, count):
    shape = checked_shape(shape)
    if not shape:
        raise ValueError("packed shape must have rank >= 1")
    if shape[-1] != row_bytes(count, bits):
        raise ValueError("packed last axis must have the exact byte count")
    checked_shape((*shape[:-1], count))
    checked_shape((*shape[:-1], shape[-1] + 1), 4)
    return shape


def _unpack(graph, source, shape, bits, count, prefix):
    def constant(suffix, value):
        return graph.constant(prefix + suffix, value)

    def node(op, inputs, suffix, **attributes):
        return graph.node(op, inputs, prefix + suffix, **attributes)

    start = constant("start", np.int64(0))
    limit = constant("count", np.int64(count))
    step = constant("step", np.int64(1))
    eight = constant("eight", np.int64(8))
    width = constant("width", np.int64(bits))
    indices = node("Range", [start, limit, step], "indices")
    whole = node("Div", [indices, eight], "whole")
    remainder = node("Mod", [indices, eight], "remainder")
    base = node("Mul", [whole, width], "base")
    tail = node("Mul", [remainder, width], "tail")
    carry = node("Div", [tail, eight], "carry")
    byte = node("Add", [base, carry], "byte")
    shift = node("Mod", [tail, eight], "shift")
    shift32 = node("Cast", [shift], "shift32", to=TensorProto.UINT32)
    next_byte = node("Add", [byte, step], "next_byte")
    pads = constant("pads", np.array([0] * len(shape) + [0] * (len(shape) - 1) + [1], np.int64))
    zero = constant("pad_zero", np.uint8(0))
    padded = node("Pad", [source, pads, zero], "padded", mode="constant")
    low = node("Gather", [padded, byte], "low", axis=-1)
    high = node("Gather", [padded, next_byte], "high", axis=-1)
    low32 = node("Cast", [low], "low32", to=TensorProto.UINT32)
    high32 = node("Cast", [high], "high32", to=TensorProto.UINT32)
    eight32 = constant("eight32", np.uint32(8))
    upper = node("BitShift", [high32, eight32], "upper", direction="LEFT")
    word = node("BitwiseOr", [low32, upper], "word")
    shifted = node("BitShift", [word, shift32], "shifted", direction="RIGHT")
    mask = constant("mask", np.uint32((1 << bits) - 1))
    masked = node("BitwiseAnd", [shifted, mask], "masked")
    unsigned = node("Cast", [masked], "unsigned", to=TensorProto.UINT8)
    logical_shape = constant("logical_shape", np.array((*shape[:-1], count), np.int64))
    return node("Reshape", [unsigned, logical_shape], "codes", allowzero=1)


def build_unpack_model(packed_shape, bits, count):
    """Build a generic unsigned unpack graph; high padding bits are ignored."""
    bits = checked_width(bits)
    count = checked_integer(count, "count")
    shape = _packed_shape(packed_shape, bits, count)
    graph = _Graph()
    output = _unpack(graph, "packed", shape, bits, count, "u_")
    return graph.model(
        [helper.make_tensor_value_info("packed", TensorProto.UINT8, shape)],
        output,
        TensorProto.UINT8,
        (*shape[:-1], count),
    )


def build_int3_decode_model(packed_shape, count, block_size, scale_dtype=TensorProto.FLOAT, explicit_zeros=False):
    """Build FLOAT INT3 decoding, including reference-only partial K blocks.

    scale_dtype is an ONNX FLOAT/FLOAT16/BFLOAT16 enum. Inputs use rank-3
    [E,N,...] layouts; no activation/routing operator or draft packed op is used.
    """
    count = checked_integer(count, "count")
    blocks = block_count(count, block_size)
    shape = _packed_shape(packed_shape, 3, count)
    if len(shape) != 3:
        raise ValueError("INT3 packed shape must have rank 3")
    scale_dtype = checked_integer(scale_dtype, "scale_dtype", 1)
    if scale_dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
        raise TypeError("scale_dtype must be FLOAT, FLOAT16, or BFLOAT16")
    if not isinstance(explicit_zeros, bool):
        raise TypeError("explicit_zeros must be bool")
    checked_shape((*shape[:-1], count), 4)
    graph = _Graph()
    inputs = [
        helper.make_tensor_value_info("packed", TensorProto.UINT8, shape),
        helper.make_tensor_value_info("scales", scale_dtype, (*shape[:-1], blocks)),
    ]
    codes = _unpack(graph, "packed", shape, 3, count, "w_")
    floating = graph.node("Cast", [codes], "float_codes", to=TensorProto.FLOAT)
    divisor = graph.constant("block_size", np.int64(block_size))
    indices = graph.node("Div", ["w_indices", divisor], "block_indices")
    scales = graph.node("Cast", ["scales"], "float_scales", to=TensorProto.FLOAT)
    expanded_scales = graph.node("Gather", [scales, indices], "expanded_scales", axis=-1)
    if explicit_zeros:
        zero_shape = _packed_shape((*shape[:-1], row_bytes(blocks, 3)), 3, blocks)
        inputs.append(helper.make_tensor_value_info("zeros", TensorProto.UINT8, zero_shape))
        zero_codes = _unpack(graph, "zeros", zero_shape, 3, blocks, "z_")
        zero_float = graph.node("Cast", [zero_codes], "float_zeros", to=TensorProto.FLOAT)
        zero = graph.node("Gather", [zero_float, indices], "expanded_zeros", axis=-1)
    else:
        zero = graph.constant("default_zero", np.float32(4))
    centered = graph.node("Sub", [floating, zero], "centered")
    output = graph.node("Mul", [centered, expanded_scales], "weights")
    return graph.model(inputs, output, TensorProto.FLOAT, (*shape[:-1], count))
