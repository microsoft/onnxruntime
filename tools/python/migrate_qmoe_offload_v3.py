#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Zero-copy ONNX graph migration tool for QMoEOffload v3 (Design Sections 5.2, 5.3, and 6 / F8, F12).

Migrates existing 26-input SharedExpertBlockV1 QMoEOffload graphs in-place or across directories
to the v3 compact signature without loading or regenerating multi-GiB external expert weights:
  - Input 21 (expert_blocks) -> Input 2
  - Input 24 (fc2_bias)      -> Input 3
  - Input 25 (token_phase)   -> Input 4 (optional)
  - Removes legacy bank inputs 2..20 and external slab/table placeholders 22..23
  - Strips optional diagnostic outputs (1..3) when --strip-diagnostics is specified
  - Stamps explicit required v3 attributes: k, hidden_size, inter_size, layout_id, layout_version,
    normalize_routing_weights, activation_alpha, activation_beta, swiglu_limit, weights_prepacked
  - Verifies representative external payload slices and in-block vs. separate fc2_bias parity
    using bounded pread reads without loading external data into RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper


V3_INT4_LAYOUT_ID = "ort.qmoe.w4a16.block32.shared.v1"
BANK_ALIGNMENT = 256


def _align_up(value: int, alignment: int = BANK_ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def compute_shared_block_layout(hidden: int, inter: int) -> dict[str, int]:
    if hidden <= 0 or inter <= 0 or hidden % 64 != 0 or inter % 64 != 0:
        raise ValueError(f"Invalid shared block dimensions: hidden={hidden}, inter={inter}")
    n1 = 2 * inter
    off = 0
    fc1_weights = off
    off += _align_up(n1 * (hidden // 2))
    fc1_scales = off
    off += _align_up((hidden // 32) * n1 * 2)
    fc1_bias = off
    off += _align_up(n1 * 2)
    fc2_weights = off
    off += _align_up(hidden * (inter // 2))
    fc2_scales = off
    off += _align_up((inter // 32) * hidden * 2)
    fc2_bias = off
    off += _align_up(hidden * 2)
    return {
        "hidden": hidden,
        "inter": inter,
        "fc1_weights": fc1_weights,
        "fc1_scales": fc1_scales,
        "fc1_bias": fc1_bias,
        "fc2_weights": fc2_weights,
        "fc2_scales": fc2_scales,
        "fc2_bias": fc2_bias,
        "stride": off,
    }


def _get_attr_map(node: onnx.NodeProto) -> dict[str, onnx.AttributeProto]:
    return {a.name: a for a in node.attribute}


def _get_int_attr(attrs: dict[str, onnx.AttributeProto], name: str, default: int | None = None) -> int | None:
    if name not in attrs:
        return default
    return int(attrs[name].i)


def _get_float_attr(attrs: dict[str, onnx.AttributeProto], name: str, default: float) -> float:
    if name not in attrs:
        return default
    return float(attrs[name].f)


def _get_str_attr(attrs: dict[str, onnx.AttributeProto], name: str, default: str) -> str:
    if name not in attrs:
        return default
    val = attrs[name].s
    return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)


def _set_int_attr(node: onnx.NodeProto, name: str, value: int) -> None:
    for i, a in enumerate(node.attribute):
        if a.name == name:
            node.attribute[i].CopyFrom(helper.make_attribute(name, int(value)))
            return
    node.attribute.append(helper.make_attribute(name, int(value)))


def _set_float_attr(node: onnx.NodeProto, name: str, value: float) -> None:
    for i, a in enumerate(node.attribute):
        if a.name == name:
            node.attribute[i].CopyFrom(helper.make_attribute(name, float(value)))
            return
    node.attribute.append(helper.make_attribute(name, float(value)))


def _set_str_attr(node: onnx.NodeProto, name: str, value: str) -> None:
    for i, a in enumerate(node.attribute):
        if a.name == name:
            node.attribute[i].CopyFrom(helper.make_attribute(name, value))
            return
    node.attribute.append(helper.make_attribute(name, value))


def _read_tensor_slice_bytes(
    tensor: onnx.TensorProto, base_dir: Path, byte_offset: int, byte_length: int
) -> bytes:
    if tensor.data_location == TensorProto.EXTERNAL:
        ext = {entry.key: entry.value for entry in tensor.external_data}
        loc = ext.get("location", "")
        if not loc:
            raise ValueError(f"External tensor '{tensor.name}' is missing location.")
        ext_offset = int(ext.get("offset", "0"))
        ext_length = int(ext.get("length", "0"))
        file_path = (base_dir / loc).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"External data file not found: {file_path}")
        if ext_length > 0 and byte_offset + byte_length > ext_length:
            raise ValueError(f"Slice [{byte_offset}, {byte_offset + byte_length}) exceeds external length {ext_length}")
        with file_path.open("rb") as f:
            f.seek(ext_offset + byte_offset)
            data = f.read(byte_length)
            if len(data) != byte_length:
                raise ValueError(f"Short read from {file_path} for tensor '{tensor.name}'")
            return data
    if tensor.raw_data:
        raw = tensor.raw_data
        if byte_offset + byte_length > len(raw):
            raise ValueError(f"Slice [{byte_offset}, {byte_offset + byte_length}) exceeds raw_data length {len(raw)}")
        return raw[byte_offset : byte_offset + byte_length]
    arr = onnx.numpy_helper.to_array(tensor)
    raw = arr.tobytes()
    return raw[byte_offset : byte_offset + byte_length]


def migrate_qmoe_offload_model(
    model_or_path: onnx.ModelProto | str | Path,
    output_path: str | Path | None = None,
    *,
    strip_diagnostics: bool = False,
    drop_phase_input: bool = False,
    preserve_relative_external_paths: bool = True,
    verify_Representative_experts: int = 3,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    src_dir: Path | None = None
    if isinstance(model_or_path, (str, Path)):
        src_path = Path(model_or_path).resolve()
        src_dir = src_path.parent
        model = onnx.load_model(str(src_path), load_external_data=False)
    else:
        model = onnx.ModelProto()
        model.CopyFrom(model_or_path)

    initializers = {init.name: init for init in model.graph.initializer}
    migrated_nodes = 0
    verified_slices = 0
    node_summaries = []

    for node_idx, node in enumerate(model.graph.node):
        if node.domain != "com.microsoft" or node.op_type != "QMoEOffload":
            continue

        attrs = _get_attr_map(node)
        inputs = list(node.input)
        is_legacy_shared = len(inputs) >= 22 and bool(inputs[21])
        is_already_v3_compact = (
            not is_legacy_shared
            and _get_int_attr(attrs, "layout_version", -1) == 1
            and len(inputs) >= 3
            and bool(inputs[2])
            and (len(inputs) <= 5 or all(not x for x in inputs[5:]))
        )
        if not is_legacy_shared and not is_already_v3_compact:
            continue

        if is_legacy_shared:
            for legacy_slot in range(2, 21):
                if legacy_slot < len(inputs) and inputs[legacy_slot]:
                    raise ValueError(
                        f"Node '{node.name}' mixes legacy bank input slot {legacy_slot} with shared expert_blocks."
                    )
            for unsupported_slot in (22, 23):
                if unsupported_slot < len(inputs) and inputs[unsupported_slot]:
                    raise ValueError(
                        f"Node '{node.name}' supplies unsupported external slab/table slot {unsupported_slot}."
                    )
            blocks_name = inputs[21]
            bias_name = inputs[24] if len(inputs) > 24 else ""
            phase_name = inputs[25] if len(inputs) > 25 else ""
            if not blocks_name or not bias_name:
                raise ValueError(f"Legacy shared QMoEOffload node '{node.name}' requires slot 21 and slot 24.")

            new_inputs = [inputs[0], inputs[1], blocks_name, bias_name]
            if phase_name and not drop_phase_input:
                new_inputs.append(phase_name)
            del node.input[:]
            node.input.extend(new_inputs)
        else:
            blocks_name = inputs[2]
            bias_name = inputs[3] if len(inputs) > 3 else ""
            phase_name = inputs[4] if len(inputs) > 4 else ""
            if drop_phase_input and len(node.input) >= 5:
                del node.input[4:]

        if strip_diagnostics and len(node.output) > 1:
            removed_outputs = list(node.output[1:])
            del node.output[1:]
            used_names = {inp for n in model.graph.node for inp in n.input}
            kept_graph_outputs = [
                out for out in model.graph.output if out.name not in removed_outputs or out.name in used_names
            ]
            del model.graph.output[:]
            model.graph.output.extend(kept_graph_outputs)

        hidden = _get_int_attr(attrs, "hidden_size", 0) or 0
        inter = _get_int_attr(attrs, "inter_size", 0) or 0
        top_k = _get_int_attr(attrs, "k", 0) or 0
        if hidden <= 0 or inter <= 0 or top_k <= 0:
            raise ValueError(
                f"QMoEOffload node '{node.name}' requires positive k={top_k}, hidden_size={hidden}, inter_size={inter}."
            )

        layout = compute_shared_block_layout(hidden, inter)
        _set_int_attr(node, "k", top_k)
        _set_int_attr(node, "hidden_size", hidden)
        _set_int_attr(node, "inter_size", inter)
        _set_int_attr(node, "layout_version", 1)
        _set_str_attr(node, "layout_id", _get_str_attr(attrs, "layout_id", V3_INT4_LAYOUT_ID))
        _set_int_attr(node, "weights_prepacked", 1)
        _set_int_attr(node, "normalize_routing_weights", _get_int_attr(attrs, "normalize_routing_weights", 0) or 0)
        _set_float_attr(node, "activation_alpha", _get_float_attr(attrs, "activation_alpha", 1.0))
        _set_float_attr(node, "activation_beta", _get_float_attr(attrs, "activation_beta", 0.0))
        if "swiglu_limit" in attrs:
            _set_float_attr(node, "swiglu_limit", _get_float_attr(attrs, "swiglu_limit", 0.0))
        if _get_int_attr(attrs, "layer_index", None) is None:
            _set_int_attr(node, "layer_index", migrated_nodes)

        # Bounded metadata and representative payload verification without loading full external data.
        if blocks_name in initializers:
            blk_init = initializers[blocks_name]
            if list(blk_init.dims)[1] != layout["stride"]:
                raise ValueError(
                    f"Initializer '{blocks_name}' stride {blk_init.dims[1]} != expected {layout['stride']}"
                )
            num_experts = int(blk_init.dims[0])
            if bias_name and bias_name in initializers and (src_dir is not None or blk_init.data_location != TensorProto.EXTERNAL):
                bias_init = initializers[bias_name]
                if list(bias_init.dims) != [num_experts, hidden]:
                    raise ValueError(
                        f"Initializer '{bias_name}' shape {list(bias_init.dims)} != [{num_experts}, {hidden}]"
                    )
                base_dir = src_dir or Path.cwd()
                row_bytes = hidden * 2
                sample_experts = sorted({0, num_experts // 2, num_experts - 1})[:verify_Representative_experts]
                for exp_id in sample_experts:
                    in_blk_off = exp_id * layout["stride"] + layout["fc2_bias"]
                    in_blk_bytes = _read_tensor_slice_bytes(blk_init, base_dir, in_blk_off, row_bytes)
                    sep_bytes = _read_tensor_slice_bytes(bias_init, base_dir, exp_id * row_bytes, row_bytes)
                    if in_blk_bytes != sep_bytes and any(b != 0 for b in in_blk_bytes):
                        raise ValueError(
                            f"Node '{node.name}' expert {exp_id} in-block fc2_bias disagrees with separate '{bias_name}'."
                        )
                    verified_slices += 1

        migrated_nodes += 1
        node_summaries.append(
            {
                "node_index": node_idx,
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "hidden_size": hidden,
                "inter_size": inter,
                "stride": layout["stride"],
            }
        )

    if drop_phase_input:
        used_inputs = {inp for n in model.graph.node for inp in n.input}
        kept_graph_inputs = [inp for inp in model.graph.input if inp.name in used_inputs]
        del model.graph.input[:]
        model.graph.input.extend(kept_graph_inputs)

    if output_path is not None:
        out_path = Path(output_path).resolve()
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if src_dir is not None and src_dir != out_dir and preserve_relative_external_paths:
            for init in model.graph.initializer:
                if init.data_location == TensorProto.EXTERNAL:
                    ext = {entry.key: entry.value for entry in init.external_data}
                    loc = ext.get("location", "")
                    if loc:
                        resolved_target = (src_dir / loc).resolve()
                        link_path = out_dir / loc
                        link_path.parent.mkdir(parents=True, exist_ok=True)
                        if not link_path.exists() and not link_path.is_symlink():
                            try:
                                os.link(resolved_target, link_path)
                            except OSError:
                                link_path.symlink_to(resolved_target)
        onnx.save_model(model, str(out_path), save_as_external_data=False)

    manifest = {
        "schema": "ort.qmoe.offload.v3.migration.v1",
        "migrated_nodes": migrated_nodes,
        "verified_bias_slices": verified_slices,
        "strip_diagnostics": strip_diagnostics,
        "drop_phase_input": drop_phase_input,
        "nodes": node_summaries,
    }
    return model, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input ONNX model path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output migrated ONNX model path.")
    parser.add_argument("--manifest", type=Path, help="Optional output JSON manifest path.")
    parser.add_argument("--strip-diagnostics", action="store_true", help="Remove optional diagnostic outputs 1..3.")
    parser.add_argument("--drop-phase-input", action="store_true", help="Omit optional token_phase input.")
    args = parser.parse_args()

    _, manifest = migrate_qmoe_offload_model(
        args.input,
        args.output,
        strip_diagnostics=args.strip_diagnostics,
        drop_phase_input=args.drop_phase_input,
    )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
