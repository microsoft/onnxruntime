import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL"
OUTPUT = Path("/workspace/qwen38-int4-hc-projection-inner-unroll/build-v1")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")
COMPONENT = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2")
SOURCE = COMPONENT / "hc_post_norm_projection.cc"
HELPER = r'''
int HCProjectionInnerUnrollMode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordHCProjectionInnerUnrollShader() {
  const char* path = std::getenv("ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"HCProjectionInnerUnrollShader\",\"mode\":"
         << HCProjectionInnerUnrollMode() << ",\"component_steps\":4}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write HC projection inner-unroll statistics to ", path);
}
'''


def program_parts(source):
    start = source.index("class HCPostNormProjectionProgram final")
    end = source.index("class HCPostNormProjectionF16", start)
    return start, end, source[start:end]


def transform_program(program):
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{HCProjectionInnerUnrollMode() == 1 ? '
                           '"HCPostNormProjectionInnerUnrollF16" : "HCPostNormProjectionF16"}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();",
                           "    RecordHCProjectionInnerUnrollShader();\n"
                           "    auto& code = shader.MainFunctionBody();")
    start = program.index('         << "  for (var i = 0u; i < 4u; i++) {\\n"')
    body_start = program.index('         << "    let low', start)
    body_end = program.index('         << "  }\\n"', body_start)
    end = body_end + len('         << "  }\\n"')
    body = program[body_start:body_end]
    replacement = ''';\n    if (HCProjectionInnerUnrollMode() == 0) {
      code << "  for (var i = 0u; i < 4u; i++) {\\n"
''' + body + '''         << "  }\\n";
    } else {
      for (uint32_t component = 0; component < 4; ++component) {
        code << "  {\\n    let i = " << component << "u;\\n";
        code
''' + body + '''         << "  }\\n";
      }
    }
    code'''
    return program[:start] + replacement + program[end:]


def transform_source(source):
    start, end, program = program_parts(source)
    source = source[:start] + HELPER + transform_program(program) + source[end:]
    return "#include <cstdlib>\n#include <fstream>\n" + source


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component_path = COMPONENT / "build-manifest.json"
    component = json.loads(component_path.read_text())
    proof_path = PROOF / "baseline.qualification.json"
    proof = json.loads(proof_path.read_text())
    commands = json.loads((PROOF / "recovered-commands.json").read_text())
    assert proof["all_152_original_objects_byte_identical"]
    assert len(commands) == len(proof["compile_commands"]) == 152

    def original_object(path):
        value = Path(path)
        return value if value.is_absolute() else Path(parent["working_directory"]) / value

    for index, original in enumerate(commands):
        assert sha(original_object(original)) == sha(PROOF / "baseline" / f"object-{index}.o"), original
    for name, digest in proof["sdk_sha256"].items():
        assert sha(PROOF / name) == digest, name
    assert sha(SOURCE) == component["source_sha256"][SOURCE.name]
    comparator_source = COMPONENT / "native_compare.cpp"
    assert sha(comparator_source) == component["source_sha256"][comparator_source.name]
    original, = [path for path, command in commands.items()
                 if command[command.index("-c") + 1] == str(SOURCE)]
    command = proof["compile_commands"][list(commands).index(original)].copy()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    obj = OUTPUT / "hc_post_norm_projection.cc.o"
    command[command.index("-o") + 1] = str(obj)
    for option in ("-MF", "-MT"):
        if option in command:
            command[command.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert sha(obj) == sha(original_object(original)), "Original HC object rebuild mismatch"
    baseline_hash = sha(obj)
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Original provider relink mismatch"
    source = OUTPUT / SOURCE.name
    source.write_text(transform_source(SOURCE.read_text()))
    command[command.index("-c") + 1] = str(source)
    command.insert(1, f"-fmacro-prefix-map={source}={SOURCE}")
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert link.count(original) == 1
    link[link.index(original)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert FLAG.encode() in library.read_bytes()
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(comparator_source.read_text())
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], check=True)
    sources = [source, SOURCE, comparator_source, component_path, comparator,
               proof_path, PROOF / "recovered-commands.json",
               *[PROOF / name for name in proof["sdk_sha256"]]]
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        all_152_original_source_objects_byte_identical=True, private_kernel_layout_only=True,
        no_object_layout_changes=True, original_hc_object_sha256=baseline_hash,
        default_enabled=False, original_compute_and_fallbacks_unchanged=True,
        process_static_selector=True, ordered_component_steps=4,
        comparator_sha256=sha(OUTPUT / "native_compare"), compile_command=command, link_command=link,
        source_sha256={os.path.relpath(path, OUTPUT): sha(path) for path in sources})
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
