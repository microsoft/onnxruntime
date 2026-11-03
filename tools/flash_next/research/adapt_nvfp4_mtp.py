import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

source = Path("/home/sayanshaw/repos/qwen_38_flash_nvfp4_engine_indexshare")
destination = Path("/home/sayanshaw/repos/qwen38-nvfp4-webgpu")
model = onnx.load(source / "mtp.onnx", load_external_data=False)
original_model_bytes = model.SerializeToString()
renames = {
    "indexshare.mode": "mode",
    "indexshare.projection_rows": "projection_rows",
    "indexshare.0.indices": "past_indices",
    "indexshare.0.counts": "past_counts",
    "indexshare.base_row_indices": "base_row_indices",
    "indexshare.range_starts": "range_starts",
    "indexshare.range_ends": "range_ends",
    "/model/layers.0/attn/PackedSparseAttentionIndexer/output_0": "present_indices",
    "/model/layers.0/attn/PackedSparseAttentionIndexer/output_1": "present_counts",
    "indexshare.0.status": "indexer_status",
}
existing_names = {name for node in model.graph.node for name in (*node.input, *node.output)}
if any(new in existing_names for new in renames.values()):
    raise ValueError("MTP name adaptation would collide with existing graph values")
for node in model.graph.node:
    for values in (node.input, node.output):
        for i, name in enumerate(values):
            values[i] = renames.get(name, name)
for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
    value.name = renames.get(value.name, value.name)
converted = []
original_initializers = {t.name: t.SerializeToString() for t in model.graph.initializer}
for tensor in model.graph.initializer:
    if tensor.data_type == TensorProto.BFLOAT16:
        if not tensor.name.endswith(".scales"):
            raise ValueError(f"Unexpected BF16 initializer {tensor.name}")
        original = numpy_helper.to_array(tensor, base_dir=str(source))
        values = original.astype(np.float32)
        if not np.array_equal(values.astype(original.dtype), original):
            raise ValueError(f"Scale conversion is not exact: {tensor.name}")
        converted.append(tensor.name)
        tensor.CopyFrom(numpy_helper.from_array(values, tensor.name))
    elif tensor.SerializeToString() != original_initializers[tensor.name]:
        raise ValueError(f"Unexpected weight change: {tensor.name}")
if len(converted) != 3:
    raise ValueError(f"Expected three MTP block-scale tensors, got {converted}")
output = destination / "mtp-webgpu.onnx"
onnx.save(model, output)
restored = onnx.load(output, load_external_data=False)
inverse = {new: old for old, new in renames.items()}
for node in restored.graph.node:
    for values in (node.input, node.output):
        for i, name in enumerate(values):
            values[i] = inverse.get(name, name)
for value in (*restored.graph.input, *restored.graph.output, *restored.graph.value_info):
    value.name = inverse.get(value.name, value.name)
for tensor in restored.graph.initializer:
    if tensor.name in converted:
        tensor.ParseFromString(original_initializers[tensor.name])
if restored.SerializeToString() != original_model_bytes:
    raise ValueError("MTP adaptation changed graph data beyond the declared reversible transformations")
config_path = destination / "genai_config.json"
config = json.loads(config_path.read_text())
mtp = config["model"]["mtp"]
mtp["filename"] = output.name
mtp["base_capacity"] = 2051
mtp["inputs"].update(past_indices="past_indices", past_counts="past_counts")
mtp["outputs"].update(
    present_indices="present_indices", present_counts="present_counts", indexer_status="indexer_status",
)
config_path.write_text(json.dumps(config, indent=2) + "\n")
manifest = json.loads((destination / "webgpu-overlay.json").read_text())
manifest["mtp_adapter"] = {
    "source_graph_sha256": hashlib.sha256((source / "mtp.onnx").read_bytes()).hexdigest(),
    "output_graph_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "renames": renames,
    "lossless_bf16_to_fp32_scales": converted,
    "weights": "Every other initializer and external-data reference is unchanged",
    "graph_operations": "All node domains, operations, attributes and edges preserved modulo the bijective rename",
    "qualification": "Persisted graph inverse-transformation is byte-identical to source; full-model GPU qualification pending",
    "schema_check_limit": "Stock ONNX checker does not recognize the source graph's ONNX-domain ORT SimplifiedLayerNormalization",
}
(destination / "webgpu-overlay.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["mtp_adapter"], indent=2))
