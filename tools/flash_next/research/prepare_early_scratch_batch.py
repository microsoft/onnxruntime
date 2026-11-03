import copy
import json
from pathlib import Path

import onnx
from onnx import helper

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/home/sayanshaw/repos")
OUTPUT = REPOS / "qwen38-int4-qmoe-early-scratch/native-v1"
SOURCE = REPOS / "qwen38-int4-qmoe-node-replay/fixtures/expert_layer0_rows1.onnx"
NODES = 32


def batch_model(original):
    model = copy.deepcopy(original)
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "QMoE":
        raise RuntimeError("Expected the sealed single-QMoE fixture")
    template = copy.deepcopy(model.graph.node[0])
    del model.graph.node[:]
    outputs = []
    for index in range(NODES):
        node = copy.deepcopy(template)
        node.name = f"qmoe_batch_{index}"
        node.output[0] = f"qmoe_output_{index}"
        model.graph.node.append(node)
        outputs.append(node.output[0])
    model.graph.node.append(helper.make_node("Concat", outputs, ["y"], axis=0, name="collect_all_outputs"))
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = NODES
    return model


def batch_benchmark(source):
    source = replace_once(source, "#include <cstdint>", "#include <cstdint>\n#include <cstring>")
    source = replace_once(source, "info.GetElementCount() != 2560", "info.GetElementCount() != 81920")
    source = replace_once(source, "      if (round >= 0) {\n        seconds.push_back",
        "      const auto* values = outputs[0].GetTensorData<Ort::Float16_t>();\n"
        "      for (size_t i = 0; i < 81920; ++i) {\n"
        "        if (!std::isfinite(values[i].ToFloat()))\n"
        "          throw std::runtime_error(\"Nonfinite batch output\");\n"
        "      }\n"
        "      for (size_t node = 1; node < 32; ++node) {\n"
        "        if (std::memcmp(values, values + node * 2560, 5120) != 0)\n"
        "          throw std::runtime_error(\"Independent batch outputs disagree\");\n"
        "      }\n"
        "      if (round >= 0) {\n        seconds.push_back")
    return replace_once(source, "i < 5120;", "i < 163840;")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = onnx.load(SOURCE, load_external_data=False)
    for tensor in original.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value != "model.onnx.data":
                raise RuntimeError("Unexpected shared external-weight location")
    (OUTPUT / "model.onnx.data").symlink_to("../../qwen38-int4-qmoe-node-replay/fixtures/model.onnx.data")
    fixture = OUTPUT / "qmoe_batch32.onnx"
    onnx.save(batch_model(original), fixture)
    onnx.checker.check_model(str(fixture))
    benchmark_source = ROOT / "fc2_private_a_benchmark.cpp"
    benchmark = OUTPUT / "native_benchmark.cpp"
    benchmark.write_text(batch_benchmark(benchmark_source.read_text()))
    (OUTPUT / "generation.json").write_text(json.dumps({
        "status": "generated_not_compiled_or_qualified",
        "qmoe_nodes": NODES,
        "concat_nodes": 1,
        "output_elements": NODES * 2560,
        "same_weight_inputs_and_routes_per_batch": True,
        "all_outputs_finite_and_bitwise_identical_within_batch_required": True,
        "readback_overhead_amortized_not_model_latency": True,
        "weight_bytes_copied": 0,
        "external_weight_alias": str(OUTPUT / "model.onnx.data"),
        "source_sha256": {str(path): sha(path) for path in
                          (Path(__file__), SOURCE, fixture, benchmark_source, benchmark)},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
