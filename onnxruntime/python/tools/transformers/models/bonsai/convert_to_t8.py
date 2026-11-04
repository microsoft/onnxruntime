#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import external_data_helper, helper, numpy_helper
from t8_codec import LAYOUT, encode_q2


def convert_model(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    data_path = output.with_name(output.name + ".data")
    if source == output or output.exists() or data_path.exists():
        raise ValueError("Choose a new output path; conversion never overwrites an existing model or data file")
    model = onnx.load(source, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    identities = {node.output[0]: node.input[0] for node in model.graph.node if node.op_type == "Identity"}

    def resolve(name: str) -> onnx.TensorProto:
        seen = set()
        while name in identities:
            if name in seen:
                raise ValueError(f"Identity cycle at {name}")
            seen.add(name)
            name = identities[name]
        if name not in initializers:
            raise ValueError(f"Expected constant initializer for {name}")
        return initializers[name]

    taken = set(initializers)
    taken.update(name for node in model.graph.node for name in (*node.input, *node.output))
    taken.update(value.name for value in model.graph.input)

    def unique_name(base: str) -> str:
        # Matrices that share only B or only scales must still get distinct T8 initializers.
        name, suffix = base, 0
        while name in taken:
            suffix += 1
            name = f"{base}_{suffix}"
        taken.add(name)
        return name

    def array(tensor: onnx.TensorProto) -> np.ndarray:
        return numpy_helper.to_array(tensor, base_dir=str(source.parent))

    output.parent.mkdir(parents=True, exist_ok=True)
    matrices = {}
    written = set()
    original_bytes = t8_bytes = count = 0
    with data_path.open("xb") as data:

        def store(tensor: onnx.TensorProto) -> None:
            raw = tensor.raw_data if tensor.HasField("raw_data") else array(tensor).tobytes()
            offset = data.tell()
            data.write(raw)
            tensor.raw_data = b""
            external_data_helper.set_external_data(tensor, data_path.name, offset=offset, length=len(raw))
            tensor.ClearField("raw_data")
            for field in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data"):
                tensor.ClearField(field)
            written.add(tensor.name)

        for node in model.graph.node:
            if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
                continue
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            if attrs.get("bits") != 2 or attrs.get("block_size") != 128 or attrs.get("accuracy_level", 0) not in (0, 1):
                raise ValueError(f"{node.name}: expected Q2/group128 with float32 computation")
            if len(node.input) < 4 or not node.input[3] or any(node.input[4:]):
                raise ValueError(f"{node.name}: expected explicit zero point and no g_idx or bias")
            weight, scales, zeros = (resolve(name) for name in node.input[1:4])
            key = (weight.name, scales.name, int(attrs["N"]), int(attrs["K"]))
            n, k = key[2:]
            if k <= 0 or k % 128:
                raise ValueError(f"{node.name}: Q2 shape disagrees with K/N")
            # Zero points are not part of the matrix cache key, so validate them for every node.
            zp = array(zeros)
            if zp.dtype != np.uint8:
                raise ValueError(f"{node.name}: expected packed uint8 zero points")
            expected_zp_bytes = (k // 128 + 3) // 4
            if zp.shape not in ((n, expected_zp_bytes), (n * expected_zp_bytes,)):
                raise ValueError(f"{node.name}: invalid zero-point shape")
            zp_codes = ((zp.reshape(n, -1, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(n, -1)
            if not np.all(zp_codes[:, : k // 128] == 1):
                raise ValueError(f"{node.name}: every zero point must equal 1")
            if key not in matrices:
                packed, s = array(weight), array(scales)
                if s.dtype != np.float32:
                    raise ValueError(f"{node.name}: MatMulTernary currently supports float32 activations only")
                if packed.shape != (n, k // 128, 32):
                    raise ValueError(f"{node.name}: Q2 shape disagrees with K/N")
                planes, t8_scales = encode_q2(packed, s, verify=True)
                names = (unique_name(weight.name + ".t8"), unique_name(scales.name + ".t8"))
                for name, value in zip(names, (planes, t8_scales), strict=True):
                    tensor = numpy_helper.from_array(value, name)
                    store(tensor)
                    initializers[name] = tensor
                matrices[key] = names
                original_bytes += packed.nbytes + s.nbytes + zp.nbytes
                t8_bytes += planes.nbytes + t8_scales.nbytes
                print(f"[{len(matrices)}] {node.name}: {n}x{k}", flush=True)
                del packed, s, zp, planes, t8_scales
            names = matrices[key]
            node.op_type = "MatMulTernary"
            del node.input[1:]
            node.input.extend(names)
            del node.attribute[:]
            node.attribute.extend(
                [
                    helper.make_attribute("K", int(attrs["K"])),
                    helper.make_attribute("N", int(attrs["N"])),
                    helper.make_attribute("format_version", 1),
                    helper.make_attribute("layout", LAYOUT),
                ]
            )
            count += 1
        if count == 0:
            raise ValueError("No compatible MatMulNBits nodes found")
        live = {value.name for value in model.graph.output}
        kept = []
        for node in reversed(model.graph.node):
            if any(name in live for name in node.output):
                kept.append(node)
                live.update(name for name in node.input if name)
        del model.graph.node[:]
        model.graph.node.extend(reversed(kept))
        tensors = [tensor for name, tensor in initializers.items() if name in live]
        # Copy every retained source tensor into the new data file; a source data file can share its
        # basename with the output data file, so the location string alone does not identify our writes.
        for tensor in tensors:
            if tensor.name not in written:
                store(tensor)
        del model.graph.initializer[:]
        model.graph.initializer.extend(tensors)
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update({"t8.format_version": "1", "t8.layout": LAYOUT, "t8.source": str(source)})
    helper.set_model_props(model, metadata)
    onnx.save_model(model, output)
    onnx.checker.check_model(str(output))
    summary = {
        "source": str(source),
        "output": str(output),
        "layout": LAYOUT,
        "nodes_converted": count,
        "unique_matrices": len(matrices),
        "round_trip_verified": True,
        "q2_matrix_bytes": original_bytes,
        "t8_matrix_bytes": t8_bytes,
        "matrix_byte_ratio": t8_bytes / original_bytes,
        "output_bytes": output.stat().st_size + data_path.stat().st_size,
    }
    output.with_suffix(".validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Losslessly re-encode a Bonsai Q2 ONNX model as T8.")
    parser.add_argument("--input", type=Path, required=True, help="Existing Q2 ONNX model.")
    parser.add_argument("--output", type=Path, required=True, help="New T8 ONNX model.")
    args = parser.parse_args()
    print(json.dumps(convert_model(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
