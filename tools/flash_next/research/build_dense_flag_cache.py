import json
import os
from pathlib import Path
import re
import subprocess

from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_FLAG_CACHE"
KERNEL = "MatMulNBitsFlagCached"
OUTPUT = Path("/workspace/qwen38-int4-dense-flag-cache/build-v1")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
SOURCE = PARENT / "matmul_nbits.cc"
HEADER = Path("/workspace/qwen38-int4-static-node-replay/build-v8/overlay/contrib_ops/webgpu/quantization/matmul_nbits.h")
PATTERN = re.compile(r'ParseEnvironmentVariableWithDefault<int>\(\s*"([^"]+)",\s*(-?\d+)\s*\)')
HELPER = r'''
class DenseFrozenFlags {
 public:
  void Enable(bool enabled) { enabled_ = enabled; }
  int Read(size_t index, const char* name, int fallback) const {
    if (statistics_path_ != nullptr) requests_.fetch_add(1, std::memory_order_relaxed);
    auto parse = [&]() {
      const int value = ParseEnvironmentVariableWithDefault<int>(name, fallback);
      if (statistics_path_ != nullptr) parses_.fetch_add(1, std::memory_order_relaxed);
      return value;
    };
    if (!enabled_) return parse();
    auto& slot = slots_.at(index);
    std::call_once(slot.once, [&]() { slot.value = parse(); });
    return slot.value;
  }
  ~DenseFrozenFlags() {
    if (statistics_path_ == nullptr) return;
    std::ofstream output(statistics_path_, std::ios::app);
    output << "{\"operator\":\"DenseFrozenFlags\",\"enabled\":" << enabled_
           << ",\"requests\":" << requests_.load() << ",\"parses\":" << parses_.load() << "}\n";
    if (!output) {
      LOGS_DEFAULT(ERROR) << "Cannot write dense flag statistics to " << statistics_path_;
    }
  }
 private:
  struct Slot { std::once_flag once; int value = 0; };
  mutable std::array<Slot, 8> slots_;
  bool enabled_ = false;
  const char* statistics_path_ = std::getenv("ORT_WEBGPU_DENSE_FLAG_CACHE_STATS_FILE");
  mutable std::atomic<size_t> requests_{0}, parses_{0};
};
'''


def compute_parts(source):
    start = source.index("Status MatMulNBits::ComputeInternal(")
    end = source.index("\n}\n", start) + 2
    return start, end, source[start:end]


def transform_source(source, header):
    _, end, original_compute = compute_parts(source)
    flags = list(dict.fromkeys(PATTERN.findall(original_compute)))
    assert len(flags) == 8 and len({name for name, _ in flags}) == 8, flags
    cached = PATTERN.sub(lambda match: (
        f'flag_cache_.Read({flags.index(match.groups())}, "{match.group(1)}", {match.group(2)})'),
        original_compute)
    cached = replace_once(cached, "MatMulNBits::ComputeInternal", f"{KERNEL}::ComputeInternal")
    start = header.index("class MatMulNBits final")
    finish = header.index("\n};", start) + 3
    declaration = header[start:finish]
    declaration = replace_once(declaration, "class MatMulNBits final", f"class {KERNEL} final")
    declaration = replace_once(declaration, "  MatMulNBits(const OpKernelInfo& info)",
                               f"  {KERNEL}(const OpKernelInfo& info)")
    ctor_end = declaration.index("\n  }")
    declaration = declaration[:ctor_end] + f'''
    const int mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(mode == 0 || mode == 1, "{FLAG} must be 0 or 1");
    flag_cache_.Enable(mode == 1);''' + declaration[ctor_end:]
    declaration = replace_once(declaration, " private:\n", " private:\n  mutable DenseFrozenFlags flag_cache_;\n")
    source = source[:end] + "\n\n" + cached + source[end:]
    marker = "namespace webgpu {\n\nnamespace {\n"
    source = replace_once(source, marker, marker + HELPER + declaration + "\n")
    source = replace_once(source, "    MatMulNBits);", f"    {KERNEL});")
    return replace_once(source, "#include <string>",
        "#include <array>\n#include <atomic>\n#include <cstdlib>\n#include <fstream>\n#include <mutex>\n#include <string>")


def transform_comparator(source):
    source = replace_once(source, "      Ort::Session session{env, argv[file], options};",
        '''      Ort::Session session{env, argv[file], options};
      Ort::Session candidate_session{env, argv[file], options};''')
    source = replace_once(source, "auto actual = session.Run(", "auto actual = candidate_session.Run(")
    source = replace_once(source,
        "        session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(), outputs.data(), outputs.size());",
        '''        Ort::Session invalid_session{env, argv[file], options};
        invalid_session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(), outputs.data(), outputs.size());''')
    return source


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    def original_object(path):
        value = Path(path)
        return value if value.is_absolute() else Path(parent["working_directory"]) / value

    proof_path = PROOF / "baseline.qualification.json"
    proof = json.loads(proof_path.read_text())
    assert proof["all_152_original_objects_byte_identical"]
    commands = json.loads((PROOF / "recovered-commands.json").read_text())
    assert len(commands) == len(proof["compile_commands"]) == 152
    for index, original in enumerate(commands):
        assert sha(original_object(original)) == sha(PROOF / "baseline" / f"object-{index}.o"), original
    for name, digest in proof["sdk_sha256"].items():
        assert sha(PROOF / name) == digest, name
    original, = [path for path, command in commands.items() if command[command.index("-c") + 1] == str(SOURCE)]
    index = list(commands).index(original)
    command = proof["compile_commands"][index].copy()
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
    source.write_text(transform_source(SOURCE.read_text(), HEADER.read_text()))
    command[command.index("-c") + 1] = str(source)
    command.insert(1, f"-fmacro-prefix-map={source}={SOURCE}")
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert link.count(original) == 1
    link[link.index(original)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert FLAG.encode() in library.read_bytes()
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(transform_comparator((PROOF / "native_compare.cpp").read_text()))
    comparator_command = ["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib", "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")]
    subprocess.run(comparator_command, check=True)
    sources = [source, SOURCE, HEADER, comparator, PROOF / "native_compare.cpp", proof_path,
               PROOF / "recovered-commands.json", *[PROOF / name for name in proof["sdk_sha256"]]]
    manifest = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        all_152_original_source_objects_byte_identical=True, private_kernel_layout_only=True,
        original_dense_object_sha256=baseline_hash, default_enabled=False, shader_and_arithmetic_unchanged=True,
        comparator_sha256=sha(OUTPUT / "native_compare"), compile_command=command, link_command=link,
        source_sha256={os.path.relpath(path, OUTPUT): sha(path) for path in sources},
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "provider_sha256")}))


if __name__ == "__main__":
    main()
