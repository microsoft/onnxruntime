import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_QMOE_ROW4_STATIC_WEIGHTS"
PROGRAM = "MatMulNBitsGroupedRows4StaticWeightsF16"
OUTPUT = Path("/workspace/qwen38-int4-qmoe-row4-static-weights/build-v1")
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
HELPER = r'''
int QmoeRow4StaticWeightsMode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_ROW4_STATIC_WEIGHTS", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_QMOE_ROW4_STATIC_WEIGHTS must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordQmoeRow4StaticWeightsShader() {
  const char* path = std::getenv("ORT_WEBGPU_QMOE_ROW4_STATIC_WEIGHTS_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"QmoeRow4StaticWeightsShader\",\"mode\":"
         << QmoeRow4StaticWeightsMode() << ",\"words\":4}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write row4 static-weight statistics to ", path);
}
'''


def program_parts(source):
    start = source.index("class GroupedPrefillRows4Program final")
    end = source.index("class MatMulInt3Program final", start)
    return start, end, source[start:end]


def static_program(program):
    program = program.replace("GroupedPrefillRows4Program", "GroupedPrefillRows4StaticWeightsProgram")
    program = replace_once(program, '"MatMulNBitsGroupedRows4F16"', f'"{PROGRAM}"')
    program = replace_once(program, '         << "      var weights: array<vec4<f16>, 8>;\\n"',
        "\n".join(f'         << "      var weights{index}: vec4<f16>;\\n"' for index in range(8)))
    for indent, next_line in ((10, "          cached_expert"), (8, "        partials")):
        prefix = '         << "' + " " * indent
        start = program.index(prefix + "for (var word = 0u; word < 4u; word++)")
        body_start = program.index("\n", start) + 1
        end = program.index('         << "' + next_line, body_start)
        closing = prefix + '}\\n"\n'
        original = program[start:end]
        assert original.endswith(closing)
        body = program[body_start:end - len(closing)]
        steps = []
        for word in range(4):
            step = body.replace("packed[word]", f"packed[{word}u]")
            step = step.replace("weights[word * 2u + 1u]", f"weights{word * 2 + 1}")
            step = step.replace("weights[word * 2u]", f"weights{word * 2}")
            step = step.replace("word * 2u", f"{word * 2}u")
            steps.append(prefix + '{\\n"\n' + step + closing)
        program = program[:start] + "".join(steps) + program[end:]
    assert "weights[" not in program and "var word" not in program
    return program


def transform_program(program):
    private = static_program(program)
    program = replace_once(program, 'Program{"MatMulNBitsGroupedRows4F16"}',
        'Program{QmoeRow4StaticWeightsMode() == 1 ? '
        f'"{PROGRAM}" : "MatMulNBitsGroupedRows4F16"' + '}')
    program = replace_once(program, "    const auto& a = shader.AddInput",
        "    RecordQmoeRow4StaticWeightsShader();\n"
        "    if (QmoeRow4StaticWeightsMode() == 1) {\n"
        "      return GroupedPrefillRows4StaticWeightsProgram{bias_}.GenerateShaderCode(shader);\n"
        "    }\n    const auto& a = shader.AddInput")
    return private + program


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


def comparator_source(source):
    return replace_once(source, "grouped-prefill-rows4-v2-native-", "row4-static-weights-v1-native-")


if __name__ == "__main__":
    comparator = SOURCE.parent / "native_compare.cpp"
    report = build(OUTPUT, SOURCE, transform_source, comparator, FLAG,
                   dict(static_decoded_weight_vectors=8, ordered_words=4, prefill_only=True))
    candidate = OUTPUT / "native_compare_variant.cpp"
    candidate.write_text(comparator_source(comparator.read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(candidate),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], check=True)
    report["comparator_sha256"] = sha(OUTPUT / "native_compare")
    report["source_sha256"][os.path.relpath(candidate, OUTPUT)] = sha(candidate)
    report["comparator_profile_prefix_only_changed"] = True
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(provider_sha256=report["provider_sha256"], comparator_profile_prefix_only_changed=True)))
