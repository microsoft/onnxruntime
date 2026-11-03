import json
import os
from pathlib import Path
import subprocess

from build_dense_flag_cache import transform_comparator
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_ORDERED_UNROLL2"
OUTPUT = Path("/workspace/qwen38-int4-dense-ordered-unroll2/build-v1")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")
SOURCE = PARENT / "matmul_nbits.cc"
TEMPLATE = Path("/workspace/qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc")
TEMPLATE_NAME = "dense_block32_unroll2_template.inc"
HELPER = r'''
int DenseOrderedUnrollMode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DENSE_ORDERED_UNROLL2", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_DENSE_ORDERED_UNROLL2 must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordDenseOrderedUnrollShader(uint32_t tile_size_k) {
  const char* path = std::getenv("ORT_WEBGPU_DENSE_ORDERED_UNROLL2_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"DenseOrderedUnrollShader\",\"mode\":"
         << DenseOrderedUnrollMode() << ",\"tile_size_k\":" << tile_size_k << "}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write dense ordered-unroll shader statistics to ", path);
}
'''


def transform_template(source):
    source = replace_once(source, "Status ApplyDenseBlock32Template(", "Status ApplyDenseBlock32Unroll2Template(")
    start = source.index("// 146 |")
    body_start = source.index("// 148 |", start)
    body_end = source.index("// 299 |", body_start)
    end = source.index("// 300 |", body_end)
    body = source[body_start:body_end]
    replacement = r'''ss << "  for (var pair_kidx = 0u; pair_kidx < uniforms.K; pair_kidx += ";
ss << 2 * __param_tile_size_k;
ss << ")\n  {\n";
for (uint32_t step = 0; step < 2; ++step) {
  if (step == 0) {
    ss << "    {\n      let kidx = pair_kidx;\n";
  } else {
    ss << "    if (pair_kidx + " << __param_tile_size_k << " < uniforms.K) {\n";
    ss << "      let kidx = pair_kidx + " << __param_tile_size_k << ";\n";
  }
''' + body + r'''  ss << "    }\n";
}
ss << "  }\n";
'''
    return source[:start] + replacement + source[end:]


def transform_source(source):
    source = replace_once(source, "#include <string>", "#include <cstdlib>\n#include <fstream>\n#include <string>")
    source = replace_once(source, '#include "dense_block32_template.inc"',
                          '#include "dense_block32_template.inc"\n'
                          f'#include "{TEMPLATE_NAME}"\n' + HELPER)
    source = replace_once(source, ': Program{"MatMulNBitsBlock32F16"},',
                          ': Program{DenseOrderedUnrollMode() == 1 ? '
                          '"MatMulNBitsBlock32F16Unroll2" : "MatMulNBitsBlock32F16"},')
    return replace_once(source, "  return ApplyDenseBlock32Template(shader, {",
                        '''  RecordDenseOrderedUnrollShader(tile_size_k);
  const auto apply = DenseOrderedUnrollMode() == 1
      ? ApplyDenseBlock32Unroll2Template : ApplyDenseBlock32Template;
  return apply(shader, {''')


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
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
    template_manifest = TEMPLATE.parent.parent / "build-v1/build-manifest.json"
    template_proof = json.loads(template_manifest.read_text())
    assert sha(TEMPLATE) == template_proof["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    original, = [path for path, command in commands.items()
                 if command[command.index("-c") + 1] == str(SOURCE)]
    command = proof["compile_commands"][list(commands).index(original)].copy()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    obj = OUTPUT / "matmul_nbits.cc.o"
    command[command.index("-o") + 1] = str(obj)
    for option in ("-MF", "-MT"):
        if option in command:
            command[command.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert sha(obj) == sha(original_object(original)), "Original dense object rebuild mismatch"
    baseline_hash = sha(obj)
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Original provider relink mismatch"
    source = OUTPUT / "matmul_nbits.cc"
    header = OUTPUT / TEMPLATE_NAME
    source.write_text(transform_source(SOURCE.read_text()))
    header.write_text(transform_template(TEMPLATE.read_text()))
    command[command.index("-c") + 1] = str(source)
    command.insert(1, f"-fmacro-prefix-map={source}={SOURCE}")
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert link.count(original) == 1
    link[link.index(original)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert FLAG.encode() in library.read_bytes()
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(transform_comparator((PROOF / "native_compare.cpp").read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], check=True)
    sources = [source, header, SOURCE, TEMPLATE, template_manifest, comparator,
               PROOF / "native_compare.cpp", proof_path, PROOF / "recovered-commands.json",
               *[PROOF / name for name in proof["sdk_sha256"]]]
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        all_152_original_source_objects_byte_identical=True, private_kernel_layout_only=True,
        no_object_layout_changes=True, original_dense_object_sha256=baseline_hash,
        default_enabled=False, original_compute_and_fallbacks_unchanged=True,
        process_static_selector=True, ordered_k_tiles_per_loop=2,
        comparator_sha256=sha(OUTPUT / "native_compare"), compile_command=command, link_command=link,
        source_sha256={os.path.relpath(path, OUTPUT): sha(path) for path in sources})
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
