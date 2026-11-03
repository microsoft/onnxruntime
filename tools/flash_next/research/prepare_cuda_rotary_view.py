import hashlib
import json
import os
from pathlib import Path
import subprocess

import onnx


source = Path("/home/sayanshaw/repos/qwen38-int4-webgpu-perf")
target = Path("/home/sayanshaw/repos/qwen38-int4-cuda-legacy-rope")
model = onnx.load(source / "model.onnx", load_external_data=False)
initializers = {tensor.name: tensor for tensor in model.graph.initializer}
indexers = [node for node in model.graph.node if node.op_type == "SparseAttentionIndexer"]
if len(indexers) != 12:
    raise RuntimeError(f"Expected twelve original indexers, found {len(indexers)}")
for node in indexers:
    if list(node.input[4:6]) != ["cos_cache", "sin_cache"]:
        raise RuntimeError(f"Unexpected rotary inputs for {node.name}")
for name in ("cos_cache", "sin_cache"):
    tensor = initializers[name]
    if list(tensor.dims) != [262144, 32]:
        raise RuntimeError(f"Unexpected rotary dimensions for {name}: {list(tensor.dims)}")
    batched = onnx.TensorProto()
    batched.CopyFrom(tensor)
    batched.name = name + "_cuda_legacy_batched"
    batched.dims.insert(0, 1)
    model.graph.initializer.append(batched)
    for node in indexers:
        node.input[4 if name == "cos_cache" else 5] = batched.name

locations = {entry.value for tensor in model.graph.initializer
             for entry in tensor.external_data if entry.key == "location"}
if locations != {"model.onnx.data", "engram.data"}:
    raise RuntimeError(f"Unexpected external locations: {sorted(locations)}")
target.mkdir(exist_ok=True)
if any(target.iterdir()):
    raise RuntimeError(f"Refusing to overwrite nonempty model directory: {target}")
links = sorted(locations | {path.name for path in source.glob("*.json")})
for name in links:
    subprocess.run(["sudo", "-n", "ln", str(source / name), str(target / name)], check=True)
    if not os.path.samefile(source / name, target / name):
        raise RuntimeError(f"External data/config is not the original inode: {name}")
onnx.save(model, target / "model.onnx")
onnx.checker.check_model(str(target / "model.onnx"))
manifest = {
    "purpose": "Batch-one correctness reference with an older compatible CUDA provider",
    "transformation": "Only indexer rotary inputs use added rank-three views of the original external bytes",
    "not_identical_graph": True,
    "source_graph_sha256": hashlib.sha256((source / "model.onnx").read_bytes()).hexdigest(),
    "adapted_graph_sha256": hashlib.sha256((target / "model.onnx").read_bytes()).hexdigest(),
    "source": str(source), "target": str(target),
    "indexers": len(indexers), "rotary_views": 2,
    "linked_files": links,
}
(target / "cuda_rotary_view_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
