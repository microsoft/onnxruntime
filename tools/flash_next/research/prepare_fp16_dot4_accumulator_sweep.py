import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
PARENT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-components/fixtures-v1")
OUTPUT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-accumulator/stress-v1")
EDGE = np.array([
    [0, -0.0, 0, -0.0], [-0.0, 0, -0.0, 0],
    [2**-24, -2**-24, 2**-14, -2**-14],
    [2**-14 - 2**-24, -2**-14 + 2**-24, 2**-24, -2**-24],
    [1, -1, -1, 1], [1, -1, 1, -1],
    [1.0009765625, -1, -1.0009765625, 1],
    [4096, -4096, -4096, 4096],
    [4096, -4094, -4096, 4094],
    [0.5, -0.5, -0.5, 0.5],
    [2**-10, -2**-10, 2**-11, -2**-11],
    [2**-20, -2**-19, 2**-18, -2**-17],
    [16, -15.9921875, -16, 15.9921875],
    [256, -255.875, -256, 255.875],
    [2048, -2047, -2048, 2047],
    [0, 1, -1, 0],
], dtype=np.float16)
OVERFLOW_EDGE = np.array([
    [-65504, 65504, 65504, -65504],
    [-65504, 65472, 65504, -65504],
    [-65472, 65504, 65472, -65504],
    [65504, -65504, -65504, 65504],
    [65472, -65504, -65472, 65504],
    [65504, -65472, -65504, 65504],
    [-0.0, 0, 0, -0.0], [0, -0.0, -0.0, 0],
    [1, -1, -1, 1], [16384, -16384, -16384, 16384],
    [32768, -32768, -32768, 32768], [61440, -61440, -61440, 61440],
    [65504, -65504, -65504, 65472],
    [2**-24, -2**-14, -2**-14, 2**-14 - 2**-24],
    [2**-14, -2**-24, -2**-24, 2**-14],
    [0, 0, 0, 0],
], dtype=np.float16)


def finite_bits():
    bits = np.arange(65536, dtype=np.uint16)
    return bits[(bits & 0x7c00) != 0x7c00]


def bounded_scales(values, maximum):
    peak = float(np.max(np.abs(values.astype(np.float32))))
    divisor = 2 ** max(0, int(np.ceil(np.log2(peak / maximum)))) if peak > 0 else 1
    result = (values.astype(np.float32) / divisor).astype(np.float16)
    if not np.all(np.isfinite(result)) or float(np.max(np.abs(result.astype(np.float32)))) > maximum:
        raise RuntimeError("Unbounded stress coefficients")
    return result


def save_fixture(template, a, b, scales, columns, destination):
    model = onnx.ModelProto()
    model.CopyFrom(template)
    tensors = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    tensors.update(a=a, b=np.concatenate([b] * (columns // 16)),
                   scales=np.concatenate([scales] * (columns // 16)))
    del model.graph.initializer[:]
    for name, values in tensors.items():
        model.graph.initializer.append(numpy_helper.from_array(values, name))
    node, = model.graph.node
    attribute, = [v for v in node.attribute if v.name == "N"]
    if attribute.i != 160:
        raise RuntimeError("Unexpected stress template")
    attribute.i = columns
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = columns
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def prepare(output=OUTPUT):
    proof_path = ROOT / "fp16-dot4-pair-components-v1-outcome.json"
    proof = json.loads(proof_path.read_text())["source_and_evidence_sha256"]
    parents = [PARENT / f"seed{seed}.onnx" for seed in range(8)]
    for path in parents:
        if sha(path) != proof[str(path)]:
            raise RuntimeError(f"Changed stress parent: {path}")
    templates = [onnx.load(path) for path in parents]
    tensors = [{v.name: numpy_helper.to_array(v) for v in m.graph.initializer} for m in templates]
    bits = finite_bits()
    if bits.size != 63488 or not np.all(np.isfinite(bits.view(np.float16))):
        raise RuntimeError("Wrong finite-half sweep")
    sweep = bits.view(np.float16).reshape(124, 8, 16, 4)
    output.mkdir(parents=True, exist_ok=False)
    batches = []
    for batch in range(125):
        directory = output / f"batch{batch:03d}"
        (directory / "reference").mkdir(parents=True)
        (directory / "candidate").mkdir()
        records = []
        for seed in range(8):
            b, scales = tensors[seed]["b"][:16], tensors[seed]["scales"][:16]
            if batch < 124:
                a = sweep[batch, seed]
                scales = bounded_scales(scales, 1 / 128)
            elif seed < 4:
                a = EDGE
                scales = bounded_scales(scales, 1 / 4)
            else:
                a = OVERFLOW_EDGE
                b = np.zeros_like(b)
                scales = np.full_like(scales, 1 / 16)
            if not np.all(np.isfinite(a)):
                raise RuntimeError("Non-finite stress input")
            paths = []
            for name, columns in (("reference", 1552), ("candidate", 160)):
                path = directory / name / f"seed{seed}.onnx"
                save_fixture(templates[seed], a, b, scales, columns, path)
                paths.append(dict(path=str(path), sha256=sha(path)))
            records.append(dict(seed=seed, fixtures=paths,
                                a_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        batches.append(dict(index=batch, kind="finite_half_sweep" if batch < 124 else "edge_cases",
                            directory=str(directory), records=records))
    report = dict(status="prepared_not_qualified", finite_input_half_patterns=63488,
                  sweep_batches=124, edge_batches=1, total_batches=125,
                  expected_comparisons_per_kind=1024000, repeats=4,
                  full_fma_operand_space_covered=False, actual_model_weights_used=False,
                  generator_sha256=sha(Path(__file__)), batches=batches,
                  parent_sha256={str(p): sha(p) for p in parents})
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = prepare()
    print(json.dumps({k: report[k] for k in ("status", "finite_input_half_patterns", "total_batches")}))
