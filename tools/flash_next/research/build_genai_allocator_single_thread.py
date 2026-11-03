import hashlib
import json
from pathlib import Path
import shlex
import subprocess


FLAG = "ORT_GENAI_WEBGPU_ALLOCATOR_SINGLE_THREAD"
SOURCE = Path("/workspace/onnxruntime-genai-webgpu-fp8-checkpoint")
BUILD = SOURCE / "build-webgpu-fp8/Release"
LEGACY = SOURCE / "build-h200-int4-legacy"
OUTPUT = Path("/workspace/qwen38-int4-genai-allocator-single-thread/build-v1")
ANCHOR = """  auto session_options = OrtSessionOptions::Create();
  std::vector<Config::ProviderOptions> provider_options_list;"""
ADDITION = f"""  if (type == DeviceType::WEBGPU) {{
    bool allocator_single_thread = false;
    GetEnv("{FLAG}", allocator_single_thread);
    if (allocator_single_thread) session_options->SetIntraOpNumThreads(1);
  }}
"""


def transform_source(original):
    include = '#include "model.h"\n'
    assert original.count(include) == 1
    assert original.count(ANCHOR) == 1
    assert FLAG not in original
    modified = original.replace(include, include + '#include "env_utils.h"\n')
    return modified.replace(ANCHOR, ANCHOR.splitlines()[0] + "\n" + ADDITION
                            + ANCHOR.splitlines()[1])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    flags_path = BUILD / "CMakeFiles/onnxruntime-genai-obj.dir/flags.make"
    link_path = BUILD / "CMakeFiles/onnxruntime-genai.dir/link.txt"
    original_path = SOURCE / "src/models/model.cpp"
    flags = {}
    for line in flags_path.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    command = shlex.split(link_path.read_text())
    model_indices = [i for i, value in enumerate(command)
                     if value.endswith("/src/models/model.cpp.o")]
    cache_indices = [i for i, value in enumerate(command)
                     if value.endswith("/src/models/io/indexer_cache.cpp.o")]
    assert len(model_indices) == len(cache_indices) == 1
    outputs = [i + 1 for i, value in enumerate(command) if value == "-o"]
    assert len(outputs) == 1
    command[cache_indices[0]] = str(LEGACY / "indexer_cache.cpp.o")

    def link(model_object, output):
        values = command.copy()
        values[model_indices[0]] = str(model_object)
        values[outputs[0]] = str(output)
        subprocess.run([*values, "-Wl,--no-undefined"], cwd=BUILD, check=True)

    parent_object = BUILD / command[model_indices[0]]
    relink = OUTPUT / "baseline-relink.so"
    link(parent_object, relink)
    assert sha(relink) == sha(LEGACY / "libonnxruntime-genai.so"), "Parent relink changed"
    compiler = ["/usr/bin/c++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"],
                *flags["CXX_FLAGS"]]
    baseline_object = OUTPUT / "model.baseline.cpp.o"
    subprocess.run([*compiler, "-c", str(original_path), "-o", str(baseline_object)],
                   cwd=BUILD, check=True)
    rebuilt = OUTPUT / "baseline-rebuilt.so"
    link(baseline_object, rebuilt)
    assert sha(rebuilt) == sha(relink), "Current source does not reproduce frozen runtime"
    original = original_path.read_text()
    snapshot = OUTPUT / "model.original.cpp"
    snapshot.write_text(original)
    generated = OUTPUT / "model.cpp"
    generated.write_text(transform_source(original))
    candidate_object = OUTPUT / "model.cpp.o"
    subprocess.run([*compiler, "-iquote", str(original_path.parent),
                    f"-fmacro-prefix-map={generated}={original_path}",
                    "-c", str(generated), "-o", str(candidate_object)], cwd=BUILD, check=True)
    runtime = OUTPUT / "libonnxruntime-genai.so"
    link(candidate_object, runtime)
    report = dict(status="built_not_qualified", flag=FLAG,
                  baseline_relink_byte_identical=True, baseline_rebuild_byte_identical=True,
                  runtime_sha256=sha(runtime), parent_runtime_sha256=sha(relink),
                  source_sha256={str(path): sha(path) for path in
                                 (Path(__file__), original_path, snapshot, generated, flags_path,
                                  link_path, parent_object, LEGACY / "indexer_cache.cpp.o")},
                  scope="WebGPU dummy allocator session only; decoder/embedding options unchanged",
                  model_and_provider_unchanged=True, default_enabled=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "runtime_sha256", "baseline_rebuild_byte_identical")}), flush=True)


if __name__ == "__main__":
    main()
