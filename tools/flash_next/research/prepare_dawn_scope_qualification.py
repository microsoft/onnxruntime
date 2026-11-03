import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/home/sayanshaw/repos")
OUTPUT = REPOS / "qwen38-int4-dawn-compute-scope-reuse/native-v1"


def fingerprint_comparator(source):
    source = replace_once(source, "#include <cmath>", "#include <cmath>\n#include <cstdint>")
    source = replace_once(source, "for (int file = 2; file < argc; ++file) {",
                          "for (int file = 2; file < argc; ++file) {\n"
                          "      uint64_t fingerprint = 14695981039346656037ull;")
    source = replace_once(source, "            ++comparisons;",
        "            for (size_t byte = 0; byte < bytes; ++byte) {\n"
        "              fingerprint ^= observed[i * bytes + byte];\n"
        "              fingerprint *= 1099511628211ull;\n"
        "            }\n            ++comparisons;")
    return replace_once(source,
        '<< " changing_rounds=128 invalid_flag_rejected=1" << std::endl;',
        '<< " changing_rounds=128 invalid_flag_rejected=1 output_fingerprint="\n'
        "                << fingerprint << std::endl;")


def batch_benchmark(source):
    source = replace_once(source, 'count != 10240', 'count != 256')
    source = replace_once(source, 'elements != 20804', 'elements != 1024')
    return source.replace("hc_operator_benchmark_completed", "scope_operator_benchmark_completed")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = REPOS / "qwen38-int4-hc-post-norm-projection/build-v2/native_compare.cpp"
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(fingerprint_comparator(original.read_text()))
    benchmark_source = ROOT / "hc_small_cta_operator_benchmark.cpp"
    benchmark = OUTPUT / "native_benchmark.cpp"
    benchmark.write_text(batch_benchmark(benchmark_source.read_text()))
    nodes = []
    outputs = []
    for chain in range(4):
        previous = "input"
        for stage in range(32):
            output = f"chain{chain}_stage{stage}"
            nodes.append(helper.make_node("MatMul", [previous, "weight"], [output],
                                          name=f"scope_chain{chain}_stage{stage}"))
            previous = output
        outputs.append(helper.make_tensor_value_info(previous, TensorProto.FLOAT16, [1, 256]))
    graph = helper.make_graph(nodes, "scope_tracking_batch",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, 256])],
        outputs, [numpy_helper.from_array(np.eye(256, dtype=np.float16) * np.float16(.96875), "weight")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    fixture = OUTPUT / "scope_batch128.onnx"
    onnx.save(model, fixture)
    manifest = {
        "status": "generated_not_compiled_or_qualified",
        "operator_graph_nodes": 128,
        "operator_graph_independent_chains": 4,
        "operator_graph_outputs": 4,
        "operator_warmups": 16,
        "operator_measured_runs": 128,
        "cross_provider_all_output_fingerprints": True,
        "source_sha256": {str(path): sha(path) for path in
                          (Path(__file__), original, comparator, benchmark_source, benchmark, fixture)},
    }
    (OUTPUT / "generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
