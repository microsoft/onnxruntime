# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""ONNX graph rewriter that converts GroupQueryAttention nodes to TurboQuant.

Usage:
    python -m onnxruntime.quantization.turboquant_kv \
        path/to/model.onnx -o path/to/model_tq.onnx \
        --preset turboquant_4bit_nc

What it does:
    1. Loads the .onnx model.
    2. Finds every GroupQueryAttention node in the com.microsoft domain.
    3. Adds the new attributes:
         kv_quant_method = "turboquant"
         key_quant_bits, value_quant_bits, norm_correction (from preset)
    4. Computes static Lloyd-Max codebook + Walsh-Hadamard matrix and injects
       them as graph initializers, wired as inputs 14 and 15 to each GQA node.
    5. Rewrites past_key_values_*/present_key_values_* tensor element types
       from fp16 to uint8, with last-dim sized to max(K_slot, V_slot) bytes.
    6. Optionally skips the first/last N attention layers (boundary protection).
    7. Saves the output model.

The model can then be loaded by ORT (with our patched libonnxruntime) and the
TurboQuant CUDA kernel will be dispatched automatically when the GQA op runs.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .centroids import solve_lloyd_max
from .hadamard import walsh_hadamard
from .quantizer import TQ_PRESETS, TurboQuantConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# GQA node identification.
# ----------------------------------------------------------------------------


def find_gqa_nodes(model: onnx.ModelProto) -> List[onnx.NodeProto]:
    """Return all GroupQueryAttention nodes in the com.microsoft domain."""
    out = []
    for node in model.graph.node:
        if node.op_type == "GroupQueryAttention" and (
            node.domain == "com.microsoft" or node.domain == ""
        ):
            out.append(node)
    return out


def get_attr(node: onnx.NodeProto, name: str, default=None):
    """Look up an attribute by name; return default if absent."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            if attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
    return default


def set_or_replace_attr(node: onnx.NodeProto, name: str, value) -> None:
    """Set or replace a single attribute on the node."""
    # Remove any existing attr with this name.
    for i, attr in enumerate(node.attribute):
        if attr.name == name:
            del node.attribute[i]
            break
    # Add the new one.
    if isinstance(value, str):
        new_attr = helper.make_attribute(name, value)
    elif isinstance(value, bool):
        new_attr = helper.make_attribute(name, int(value))
    elif isinstance(value, int):
        new_attr = helper.make_attribute(name, value)
    elif isinstance(value, float):
        new_attr = helper.make_attribute(name, value)
    else:
        raise ValueError(f"unsupported attribute type: {type(value)}")
    node.attribute.append(new_attr)


# ----------------------------------------------------------------------------
# Initializer injection.
# ----------------------------------------------------------------------------


def make_codebook_initializer(name: str, head_dim: int, bits: int, dtype) -> onnx.TensorProto:
    """Build a graph initializer holding the static Lloyd-Max codebook."""
    centroids, _ = solve_lloyd_max(head_dim, bits)
    centroids = centroids.astype(dtype)
    return numpy_helper.from_array(centroids, name=name)


def make_hadamard_initializer(name: str, head_dim: int, dtype) -> onnx.TensorProto:
    """Build a graph initializer holding the Walsh-Hadamard rotation matrix."""
    H = walsh_hadamard(head_dim, dtype=np.float32).astype(dtype)
    return numpy_helper.from_array(H, name=name)


def numpy_dtype_for_onnx(elem_type: int):
    """Map ONNX TensorProto type to numpy dtype."""
    if elem_type == TensorProto.FLOAT16:
        return np.float16
    if elem_type == TensorProto.BFLOAT16:
        # NumPy doesn't natively support bf16; ONNX spec permits raw_data.
        # We fall back to fp16 and let ORT handle the conversion (acceptable
        # because both the codebook and Hadamard are static/small).
        return np.float16
    if elem_type == TensorProto.FLOAT:
        return np.float32
    raise ValueError(f"unsupported element type for codebook init: {elem_type}")


# ----------------------------------------------------------------------------
# Type rewriting for past_key_values / present_key_values.
# ----------------------------------------------------------------------------


def find_value_info(model: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    """Look up a value_info by name across graph inputs/outputs/value_info."""
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == name:
            return vi
    return None


def slot_bytes(head_dim: int, bits: int, is_value: bool) -> int:
    """K slot: ceil(D*bits/8) + 2.  V slot: ceil(D*bits/8) + 4."""
    return (head_dim * bits + 7) // 8 + (4 if is_value else 2)


# ----------------------------------------------------------------------------
# Main rewrite.
# ----------------------------------------------------------------------------


def rewrite_model_for_turboquant(
    model: onnx.ModelProto,
    preset: str,
    boundary_n: int = 2,
) -> onnx.ModelProto:
    """Rewrite a model in place to use TurboQuant on its GQA nodes.

    Args:
        model: input ONNX model (modified in-place + returned).
        preset: TQ_PRESETS key, e.g. 'turboquant_4bit_nc'.
        boundary_n: number of first/last attention layers to leave at fp16.

    Returns:
        The same model object, modified.
    """
    if preset not in TQ_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; valid: {list(TQ_PRESETS)}")
    p = TQ_PRESETS[preset]

    gqa_nodes = find_gqa_nodes(model)
    if not gqa_nodes:
        raise RuntimeError("no GroupQueryAttention nodes found in model")
    logger.info("Found %d GroupQueryAttention nodes", len(gqa_nodes))

    n_layers = len(gqa_nodes)
    skip_idxs = set()
    if boundary_n > 0 and n_layers > 2 * boundary_n:
        skip_idxs.update(range(boundary_n))
        skip_idxs.update(range(n_layers - boundary_n, n_layers))

    # Determine head_dim from the first non-skipped node by inspecting its
    # past_key tensor shape (input slot 3). If shape inference info is missing
    # we fall back to a CLI-provided value.
    head_dim = None
    for node in gqa_nodes:
        if len(node.input) > 3 and node.input[3]:
            vi = find_value_info(model, node.input[3])
            if vi is not None and vi.type.tensor_type.shape.dim:
                dims = vi.type.tensor_type.shape.dim
                # past_key shape: (batch, num_kv_heads, max_seq, head_size).
                if len(dims) >= 4 and dims[3].dim_value > 0:
                    head_dim = dims[3].dim_value
                    break
    if head_dim is None:
        raise RuntimeError(
            "Could not infer head_dim from any GQA node's past_key shape. "
            "Run the model through onnx.shape_inference first or pass --head-dim."
        )
    logger.info("Detected head_dim=%d", head_dim)

    # Build single shared codebook + Hadamard initializers (one set per (head_dim, bits)).
    # All non-skipped nodes share these (the codebook is static, the Hadamard is
    # deterministic given head_dim).
    cb_dtype = np.float16   # we always emit fp16 initializers; ORT casts as needed
    cb_name = f"__turboquant_kcodebook__hd{head_dim}_b{p['key_quant_bits']}"
    h_name = f"__turboquant_hadamard__hd{head_dim}"

    existing_inits = {init.name for init in model.graph.initializer}
    if cb_name not in existing_inits:
        model.graph.initializer.append(
            make_codebook_initializer(cb_name, head_dim, p["key_quant_bits"], cb_dtype)
        )
    if h_name not in existing_inits:
        model.graph.initializer.append(
            make_hadamard_initializer(h_name, head_dim, cb_dtype)
        )

    # Slot byte sizes for past/present cache shape rewrite.
    k_slot = slot_bytes(head_dim, p["key_quant_bits"], is_value=False)
    v_slot = slot_bytes(head_dim, p["value_quant_bits"], is_value=True)
    cache_last_dim = max(k_slot, v_slot)
    logger.info(
        "TurboQuant slot bytes: K=%d, V=%d, cache_last_dim=%d (vs fp16 %d)",
        k_slot, v_slot, cache_last_dim, head_dim * 2,
    )

    n_rewritten = 0
    for idx, node in enumerate(gqa_nodes):
        if idx in skip_idxs:
            logger.info("Skipping layer %d (boundary protection)", idx)
            continue

        # Set new attributes.
        set_or_replace_attr(node, "kv_quant_method", "turboquant")
        set_or_replace_attr(node, "key_quant_bits", p["key_quant_bits"])
        set_or_replace_attr(node, "value_quant_bits", p["value_quant_bits"])
        set_or_replace_attr(node, "norm_correction", int(p["norm_correction"]))

        # Wire codebook + hadamard as inputs 14, 15.
        # Pad input list with empty strings if needed.
        while len(node.input) < 14:
            node.input.append("")
        if len(node.input) == 14:
            node.input.append(cb_name)
        else:
            node.input[14] = cb_name
        if len(node.input) == 15:
            node.input.append(h_name)
        else:
            node.input[15] = h_name

        # Rewrite past_key, past_value, present_key, present_value tensor types
        # to uint8 with last-dim = cache_last_dim. This requires updating the
        # value_info / graph input / graph output entries.
        for slot in (3, 4):  # past_key, past_value
            if slot < len(node.input) and node.input[slot]:
                _rewrite_cache_tensor(model, node.input[slot], cache_last_dim)
        for slot in (1, 2):  # present_key, present_value
            if slot < len(node.output) and node.output[slot]:
                _rewrite_cache_tensor(model, node.output[slot], cache_last_dim)

        n_rewritten += 1

    logger.info("Rewrote %d / %d GQA nodes to TurboQuant", n_rewritten, n_layers)
    return model


def _rewrite_cache_tensor(model: onnx.ModelProto, name: str, new_last_dim: int) -> None:
    """Rewrite a graph value_info/input/output entry to uint8 with new last dim.

    The shape becomes [B, H_kv, max_seq, new_last_dim].
    """
    for vi_list in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in vi_list:
            if vi.name == name:
                tt = vi.type.tensor_type
                tt.elem_type = TensorProto.UINT8
                if tt.shape.dim:
                    # Reset last dim.
                    last = tt.shape.dim[-1]
                    last.dim_value = new_last_dim
                    last.ClearField("dim_param")
                return


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="onnxruntime.quantization.turboquant_kv",
        description="Rewrite a model.onnx to use TurboQuant KV cache compression.",
    )
    parser.add_argument("input", help="Path to input model.onnx")
    parser.add_argument("-o", "--output", required=True, help="Path to output model_tq.onnx")
    parser.add_argument(
        "--preset",
        choices=list(TQ_PRESETS),
        default="turboquant_4bit_nc",
        help="TurboQuant preset (default: turboquant_4bit_nc).",
    )
    parser.add_argument(
        "--boundary-n",
        type=int,
        default=2,
        help="Skip first/last N attention layers (kept fp16 for accuracy). Default 2.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run onnx.checker.check_model on the output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    logger.info("Loading %s", args.input)
    model = onnx.load(args.input)

    rewrite_model_for_turboquant(
        model,
        preset=args.preset,
        boundary_n=args.boundary_n,
    )

    logger.info("Saving %s", args.output)
    onnx.save(model, args.output)

    if args.check:
        try:
            onnx.checker.check_model(args.output)
            logger.info("onnx.checker.check_model: PASS")
        except onnx.checker.ValidationError as e:
            logger.error("onnx.checker.check_model: FAIL\n%s", e)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
