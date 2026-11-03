import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from build_qmoe_node_replay import sha


def mixed_model(original):
    assert len(original.graph.node) == len(original.graph.input) == len(original.graph.output) == 1
    assert original.graph.node[0].op_type == "MatMulNBits"
    model = copy.deepcopy(original)
    activation = original.graph.input[0].name
    first = copy.deepcopy(original.graph.node[0])
    first.input[0], first.output[0], first.name = "mixed_shifted", "mixed_mid", "mixed_first"
    second = copy.deepcopy(first)
    second.input[0], second.output[0], second.name = "mixed_mid_shifted", "mixed_final", "mixed_second"
    mid, final = copy.deepcopy(original.graph.output[0]), copy.deepcopy(original.graph.output[0])
    mid.name, final.name = "mixed_mid", "mixed_final"
    shift = numpy_helper.from_array(np.asarray(0.0078125, dtype=np.float16), "mixed_shift")
    graph = helper.make_graph([
        helper.make_node("Add", [activation, "mixed_shift"], ["mixed_shifted"], name="mixed_prefix"),
        first,
        helper.make_node("Add", ["mixed_mid", "mixed_shift"], ["mixed_mid_shifted"], name="mixed_between"),
        second], "mixed_deferred_and_ready_replay", list(original.graph.input), [mid, final],
        [*original.graph.initializer, shift])
    model.graph.CopyFrom(graph)
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", type=Path, default=Path("/home/sayanshaw/repos"))
    args = parser.parse_args()
    output = args.repos / "qwen38-int4-immediate-node-replay/mixed-fixtures-v1"
    output.mkdir(parents=True, exist_ok=False)
    source_hashes, fixture_hashes = {}, {}
    for rows in (1, 2, 16):
        source = args.repos / f"qwen38-int4-generic-matmul-replay/fixtures-v1/matmul_k2560_n2560_rows{rows}.onnx"
        model = mixed_model(onnx.load(source))
        fixture = output / f"mixed_replay_rows{rows}.onnx"
        onnx.save(model, fixture)
        source_hashes[source.name], fixture_hashes[fixture.name] = sha(source), sha(fixture)
    report = dict(status="prepared_not_qualified", rows=[1, 2, 16], run_rows=[1, 2, 16, 1], width=2560,
                  outputs_per_fixture=2, rounds_per_fixture=128,
                  finite_bitwise_comparisons=2560 * (1 + 2 + 16 + 1) * 2 * 128,
                  source_fixture_sha256=source_hashes, fixture_sha256=fixture_hashes,
                  generator_sha256=sha(Path(__file__)))
    (output / "fixtures-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()
