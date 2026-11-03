import json
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import helper, numpy_helper

from build_qmoe_node_replay import sha
from prepare_fp16_dot4_accumulator_sweep import OUTPUT as DATASET, PARENT


ROOT = Path(__file__).parent
OUTPUT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-accumulator/dynamic-stress-v1")


def frame_bytes(path):
    values = {v.name: numpy_helper.to_array(v) for v in onnx.load(path).graph.initializer}
    a, b, scales = values["a"], values["b"][:16], values["scales"][:16]
    if a.dtype != np.float16 or a.shape != (16, 4) or b.dtype != np.uint8 or b.shape != (16, 1, 16) or scales.dtype != np.float16 or scales.shape != (16, 1):
        raise RuntimeError("Unexpected frame shape or dtype")
    payload = a.tobytes() + b.tobytes() + scales.tobytes()
    if len(payload) != 416 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(scales)):
        raise RuntimeError("Invalid frame data")
    return payload, a.view(np.uint16).reshape(-1)


def dynamic_model(template, columns, destination):
    model = onnx.ModelProto()
    model.CopyFrom(template)
    node, = model.graph.node
    if list(node.input) != ["a", "b", "scales"]:
        raise RuntimeError("Unexpected dynamic input names")
    del model.graph.initializer[:]
    del model.graph.input[:]
    model.graph.input.extend([
        helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT16, [16, 4]),
        helper.make_tensor_value_info("b", onnx.TensorProto.UINT8, [columns, 1, 16]),
        helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT16, [columns, 1])])
    attribute, = [v for v in node.attribute if v.name == "N"]
    attribute.i = columns
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = columns
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def prepare():
    dataset_path = DATASET / "manifest.json"
    dataset = json.loads(dataset_path.read_text())
    if dataset["total_batches"] != 125 or dataset["finite_input_half_patterns"] != 63488:
        raise RuntimeError("Unexpected prepared dataset")
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    source_paths = []
    frames, finite_patterns = [], []
    for batch in dataset["batches"]:
        for record in batch["records"]:
            value = record["fixtures"][1]
            path = Path(value["path"])
            if sha(path) != value["sha256"]:
                raise RuntimeError(f"Changed prepared fixture: {path}")
            payload, bits = frame_bytes(path)
            frames.append(payload)
            if batch["index"] < 124:
                finite_patterns.append(bits)
            source_paths.append(path)
    if len(frames) != 1000 or np.unique(np.concatenate(finite_patterns)).size != 63488:
        raise RuntimeError("Incomplete finite pattern coverage")
    micro_paths = [PARENT / f"seed{seed}.onnx" for seed in range(8)]
    qualified = json.loads((ROOT / "fp16-dot4-pair-components-v1-outcome.json").read_text())["source_and_evidence_sha256"]
    for path in micro_paths:
        if sha(path) != qualified[str(path)]:
            raise RuntimeError(f"Changed micro fixture: {path}")
    for name, data in (("micro", [frame_bytes(p)[0] for p in micro_paths]), ("sweep", frames)):
        with (output / f"{name}.f16s").open("xb") as stream:
            stream.write(struct.pack("<4sII", b"F16S", 1, len(data)))
            for payload in data:
                stream.write(payload)
    template = onnx.load(micro_paths[0])
    for name, columns in (("reference", 1552), ("candidate", 160)):
        dynamic_model(template, columns, output / f"{name}.onnx")
    paths = [output / f"{name}.f16s" for name in ("micro", "sweep")]
    paths.extend(output / f"{name}.onnx" for name in ("reference", "candidate"))
    report = dict(status="prepared_not_qualified", finite_half_patterns=63488, sweep_frames=1000,
                  micro_frames=8, frame_bytes=416, changing_gpu_inputs=True,
                  source_sha256={str(p): sha(p) for p in [Path(__file__), dataset_path, *source_paths, *micro_paths]},
                  output_sha256={str(p): sha(p) for p in paths})
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = prepare()
    print(json.dumps({k: report[k] for k in ("status", "finite_half_patterns", "sweep_frames")}))
