import json
from pathlib import Path

import onnx

from build_qmoe_node_replay import sha
from prepare_generic_matmul_replay import SHAPES
from qualify_quickgelu_mul_long_context import ROOT


OUTPUT = Path("/home/sayanshaw/repos/qwen38-int4-dense-output-shape/rank-fixtures-v1")


def set_shape(value, dimensions):
    del value.type.tensor_type.shape.dim[:]
    for size in dimensions:
        value.type.tensor_type.shape.dim.add().dim_value = size


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = ROOT / "generic-matmul-replay-v1-native-gpu0-20261029.qualification.json"
    proof = json.loads(original.read_text())
    directory = Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/fixtures-v1")
    records, selected = [], set()
    for path in sorted(directory.glob("*.onnx")):
        assert sha(path) == proof["fixture_sha256"][path.name]
        model = onnx.load(path)
        assert len(model.graph.input) == len(model.graph.output) == 1
        dimensions = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        if dimensions[0] != 1:
            continue
        node = next(node for node in model.graph.node if node.op_type == "MatMulNBits")
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        k, n = attributes["K"], attributes["N"]
        assert (k, n) in SHAPES and (k, n) not in selected
        selected.add((k, n))
        for rank in (1, 2, 3, 4):
            input_shape = [1] * (rank - 1) + [k]
            output_shape = [1] * (rank - 1) + [n]
            set_shape(model.graph.input[0], input_shape)
            set_shape(model.graph.output[0], output_shape)
            onnx.checker.check_model(model)
            target = OUTPUT / f"k{k}-n{n}-rank{rank}.onnx"
            onnx.save_model(model, target)
            records.append(dict(path=str(target), sha256=sha(target), K=k, N=n, rank=rank,
                                input_shape=input_shape, output_shape=output_shape,
                                original_fixture_sha256=sha(path)))
    assert selected == set(SHAPES) and len(records) == 36
    report = dict(status="synthetic_rank_fixtures_prepared", fixtures=records,
                  finite_bitwise_comparisons=sum(r["N"] for r in records) * 128,
                  source_sha256={str(original): sha(original), str(Path(__file__)): sha(Path(__file__))})
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(fixtures=len(records), comparisons=report["finite_bitwise_comparisons"])))


if __name__ == "__main__":
    main()
