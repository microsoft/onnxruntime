"""Replace static per-tensor UINT16 DQ constants with equivalent float initializers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def materialize(source: Path, output: Path, expected_count: int | None = None) -> int:
    source, output = source.resolve(), output.resolve()
    sidecar = output.with_name(output.name + ".data")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.parent != source.parent:
        raise ValueError("output must be beside the source to reuse its external INT4 weights")
    if output == source or output.exists() or sidecar.exists():
        raise FileExistsError(f"output model or sidecar already exists: {output}, {sidecar}")

    model = onnx.load(source, load_external_data=False)
    if any(node.op_type == "QuantizeLinear" for node in model.graph.node):
        raise ValueError("source still has activation QuantizeLinear nodes; bypass them first")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    replacements = []
    converted = set()
    for index, node in enumerate(model.graph.node):
        if node.op_type != "DequantizeLinear":
            continue
        if len(node.input) != 3 or node.input[0] not in initializers:
            raise ValueError(f"{node.name}: expected a DQ of a static initializer with explicit scale and zero point")
        value = initializers[node.input[0]]
        if value.data_type == onnx.TensorProto.INT4:
            continue
        if value.data_type != onnx.TensorProto.UINT16 or len(node.output) != 1 or node.attribute:
            raise ValueError(f"{node.name}: only per-tensor UINT16 DQ constants can be materialized")
        scale = initializers.get(node.input[1])
        zero_point = initializers.get(node.input[2])
        if scale is None or zero_point is None:
            raise ValueError(f"{node.name}: missing constant scale or zero point")
        values = numpy_helper.to_array(value, base_dir=str(source.parent))
        scales = numpy_helper.to_array(scale, base_dir=str(source.parent))
        zeros = numpy_helper.to_array(zero_point, base_dir=str(source.parent))
        if scales.shape != () or scales.dtype != np.float32 or not np.isfinite(scales) or scales <= 0:
            raise ValueError(f"{node.name}: expected a positive scalar float32 scale")
        if zeros.shape != () or zeros.dtype != np.uint16:
            raise ValueError(f"{node.name}: expected a scalar uint16 zero point")
        output_name = node.output[0]
        if output_name in initializers:
            raise ValueError(f"{node.name}: output {output_name} already names an initializer")
        floats = (values.astype(np.int32) - int(zeros)).astype(np.float32) * scales
        replacements.append(numpy_helper.from_array(floats, output_name))
        converted.add(index)
    if not converted or (expected_count is not None and len(converted) != expected_count):
        raise ValueError(f"materialized {len(converted)} UINT16 DQs, expected {expected_count or 'at least one'}")

    remaining = [node for index, node in enumerate(model.graph.node) if index not in converted]
    del model.graph.node[:]
    model.graph.node.extend(remaining)
    used_inputs = {name for node in remaining for name in node.input if name}
    used_inputs.update(value.name for value in model.graph.output)
    retained = [tensor for tensor in model.graph.initializer if tensor.name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    model.graph.initializer.extend(replacements)
    if any(
        node.op_type == "DequantizeLinear" and
        initializers[node.input[0]].data_type != onnx.TensorProto.INT4
        for node in model.graph.node
    ):
        raise ValueError("unexpected non-INT4 DQ remains after materialization")
    model.metadata_props.add(key="static_uint16_dq_materialized", value=str(len(converted)))
    # ORT shape inference must read small shape initializers inline, not from external data.
    onnx.save_model(
        model, output, save_as_external_data=True, all_tensors_to_one_file=True,
        location=sidecar.name, size_threshold=1 << 20,
    )
    if model.ir_version <= onnx.IR_VERSION:
        onnx.checker.check_model(str(output))
    else:
        print(f"Skipped ONNX checker: model IR {model.ir_version} exceeds checker IR {onnx.IR_VERSION}")
    print(f"Saved {output}: {len(converted)} static UINT16 DQs materialized; INT4 external weights retained")
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, help="Require an exact number of static UINT16 DQ nodes.")
    args = parser.parse_args()
    if args.expect_count is not None and args.expect_count <= 0:
        parser.error("--expect-count must be positive")
    materialize(args.source, args.output, args.expect_count)


if __name__ == "__main__":
    main()
