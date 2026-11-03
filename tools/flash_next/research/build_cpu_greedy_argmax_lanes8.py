import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
SOURCE = Path("/workspace/onnxruntime-genai-webgpu-fp8-checkpoint")
BUILD = SOURCE / "build-webgpu-fp8/Release"
LEGACY = SOURCE / "build-h200-int4-legacy"
OUTPUT = Path("/workspace/qwen38-int4-cpu-greedy-argmax/build-v1")
FLAG = "ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8"
ARGMAX = r'''
int32_t ArgMaxCpuLanes8(std::span<const float> values) {
  if (values.empty()) throw std::runtime_error("ArgMax requires non-empty values");
  float maxima[8];
  int32_t indices[8]{};
  // Seeding every lane from index zero retains the original sticky first-NaN behavior.
  std::fill_n(maxima, 8, values[0]);
  size_t i = 1;
  for (; values.size() - i >= 8; i += 8) {
    for (size_t lane = 0; lane < 8; ++lane) {
      const float value = values[i + lane];
      const bool greater = value > maxima[lane];
      maxima[lane] = greater ? value : maxima[lane];
      indices[lane] = greater ? static_cast<int32_t>(i + lane) : indices[lane];
    }
  }
  float best = values[0];
  int32_t index = 0;
  for (size_t lane = 0; lane < 8; ++lane) {
    if (maxima[lane] > best || (maxima[lane] == best && indices[lane] < index)) {
      best = maxima[lane];
      index = indices[lane];
    }
  }
  for (; i < values.size(); ++i) {
    if (values[i] > best) {
      best = values[i];
      index = static_cast<int32_t>(i);
    }
  }
  return index;
}
'''
SELECTOR = f'''
bool CpuGreedyArgMaxLanes8Enabled() {{
  static const bool enabled = []() {{
    bool value = false;
    GetEnv("{FLAG}", value);
    return value;
  }}();
  return enabled;
}}
'''


def transform_source(original):
    original = replace_once(original, '#include "search.h"\n',
                            '#include "search.h"\n#include "models/env_utils.h"\n')
    original = replace_once(original, "namespace Generators {\n",
                            "namespace Generators {\nnamespace {\n" + ARGMAX + SELECTOR + "}\n")
    return replace_once(original, "    auto const token = ArgMax<float>(scores);",
                        "    auto const token = CpuGreedyArgMaxLanes8Enabled()\n"
                        "                           ? ArgMaxCpuLanes8(scores)\n"
                        "                           : ArgMax<float>(scores);")


def original_argmax(header):
    return header[header.index("template <typename T, typename Score>"):
                  header.index("struct Search :")]


if __name__ == "__main__":
    preflight_path = OUTPUT / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    assert preflight["baseline_relink_byte_identical"] and preflight["baseline_search_rebuild_byte_identical"]
    source = SOURCE / "src/search.cpp"
    header = SOURCE / "src/search.h"
    assert sha(source) == preflight["source_sha256"] and sha(header) == preflight["header_sha256"]
    parent_runtime = LEGACY / "libonnxruntime-genai.so"
    assert sha(parent_runtime) == preflight["runtime_sha256"]
    assert sha(OUTPUT / "baseline-relink.so") == sha(parent_runtime)
    assert sha(OUTPUT / "baseline-search-rebuilt.so") == sha(parent_runtime)
    target = OUTPUT / "search.cpp"
    with target.open("x") as stream:
        stream.write(transform_source(source.read_text()))
    obj = OUTPUT / "search.cpp.o"
    command = [*preflight["compile_command"], "-iquote", str(source.parent),
               f"-fmacro-prefix-map={target}={source}", "-c", str(target), "-o", str(obj)]
    subprocess.run(command, cwd=BUILD, check=True)
    link = preflight["link_command"].copy()
    index = link.index(str(OUTPUT / "search.baseline.cpp.o"))
    link[index] = str(obj)
    runtime = OUTPUT / "libonnxruntime-genai.so"
    link[link.index("-o") + 1] = str(runtime)
    subprocess.run([*link, "-Wl,--no-undefined"], cwd=BUILD, check=True)
    assert FLAG.encode() in runtime.read_bytes()
    comparator_source = ROOT / "cpu_greedy_argmax_compare.cpp.in"
    comparator = OUTPUT / "cpu_greedy_argmax_compare.cpp"
    text = comparator_source.read_text()
    text = replace_once(text, "@@ORIGINAL_ARGMAX@@", original_argmax(header.read_text()))
    text = replace_once(text, "@@LANES8_ARGMAX@@", ARGMAX)
    with comparator.open("x") as stream:
        stream.write(text)
    executable = OUTPUT / "cpu_greedy_argmax_compare"
    subprocess.run(["c++", "-std=c++20", "-O3", str(comparator), "-o", str(executable)], check=True)
    evidence = [Path(__file__), comparator_source, comparator, executable, source, header, target,
                preflight_path, parent_runtime, OUTPUT / "search.baseline.cpp.o",
                LEGACY / "indexer_cache.cpp.o",
                BUILD / "CMakeFiles/onnxruntime-genai-obj.dir/flags.make",
                BUILD / "CMakeFiles/onnxruntime-genai.dir/link.txt",
                SOURCE / "src/models/env_utils.h"]
    report = dict(status="built_not_qualified", flag=FLAG, default_enabled=False,
        process_static_selector=True, runtime_sha256=sha(runtime),
        parent_runtime_sha256=sha(parent_runtime), baseline_relink_byte_identical=True,
        baseline_search_rebuild_byte_identical=True, public_object_layouts_unchanged=True,
        model_provider_and_sampling_unchanged=True, compile_command=command, link_command=link,
        source_sha256={str(path): sha(path) for path in evidence})
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], runtime_sha256=report["runtime_sha256"])), flush=True)
