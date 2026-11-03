import json
from pathlib import Path
import subprocess

from build_cpu_greedy_argmax_lanes8 import (
    ARGMAX as LANES8, BUILD, LEGACY, ROOT, SOURCE, original_argmax,
    transform_source as lanes8_source,
)
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512"
OUTPUT = Path("/workspace/qwen38-int4-cpu-greedy-argmax-avx512/build-v1")
PARENT = Path("/workspace/qwen38-int4-cpu-greedy-argmax/build-v1")
ARGMAX = r'''
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
bool CpuArgMaxAvx512Available() {
  return __builtin_cpu_supports("avx512f");
}

__attribute__((target("avx512f")))
int32_t ArgMaxCpuAvx512(std::span<const float> values) {
  if (values.empty()) throw std::runtime_error("ArgMax requires non-empty values");
  if (values.size() > static_cast<size_t>(INT32_MAX))
    throw std::runtime_error("ArgMax input exceeds int32 index range");
  __m512 maxima[2]{_mm512_set1_ps(values[0]), _mm512_set1_ps(values[0])};
  __m512i indices[2]{_mm512_setzero_si512(), _mm512_setzero_si512()};
  const __m512i offsets = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  size_t i = 1;
  for (; values.size() - i >= 32; i += 32) {
    for (size_t group = 0; group < 2; ++group) {
      const __m512 value = _mm512_loadu_ps(values.data() + i + group * 16);
      const __mmask16 greater = _mm512_cmp_ps_mask(value, maxima[group], _CMP_GT_OQ);
      maxima[group] = _mm512_mask_mov_ps(maxima[group], greater, value);
      const __m512i positions = _mm512_add_epi32(
          _mm512_set1_epi32(static_cast<int32_t>(i + group * 16)), offsets);
      indices[group] = _mm512_mask_mov_epi32(indices[group], greater, positions);
    }
  }
  float lane_maxima[32];
  int32_t lane_indices[32];
  for (size_t group = 0; group < 2; ++group) {
    _mm512_storeu_ps(lane_maxima + group * 16, maxima[group]);
    _mm512_storeu_si512(lane_indices + group * 16, indices[group]);
  }
  float best = values[0];
  int32_t index = 0;
  for (size_t lane = 0; lane < 32; ++lane) {
    if (lane_maxima[lane] > best ||
        (lane_maxima[lane] == best && lane_indices[lane] < index)) {
      best = lane_maxima[lane];
      index = lane_indices[lane];
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
#else
bool CpuArgMaxAvx512Available() { return false; }
int32_t ArgMaxCpuAvx512(std::span<const float>) {
  throw std::runtime_error("AVX512 argmax requires supported x86 compiler and CPU");
}
#endif
'''
SELECTOR = f'''
bool CpuGreedyArgMaxAvx512Enabled() {{
  static const bool enabled = []() {{
    bool value = false;
    GetEnv("{FLAG}", value);
    if (value && !CpuArgMaxAvx512Available())
      throw std::runtime_error("{FLAG} requires AVX512F CPU support");
    return value;
  }}();
  return enabled;
}}
'''


def transform_source(original):
    source = lanes8_source(original)
    source = replace_once(source, '#include "search.h"\n',
        '#include "search.h"\n#include <cstdint>\n'
        '#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))\n'
        '#include <immintrin.h>\n#endif\n')
    source = replace_once(source, "namespace Generators {\nnamespace {\n",
                          "namespace Generators {\nnamespace {\n" + ARGMAX + SELECTOR)
    return replace_once(source, "    auto const token = CpuGreedyArgMaxLanes8Enabled()",
        "    auto const token = CpuGreedyArgMaxAvx512Enabled()\n"
        "                           ? ArgMaxCpuAvx512(scores)\n"
        "                           : CpuGreedyArgMaxLanes8Enabled()")


def comparator_source(header):
    text = (ROOT / "cpu_greedy_argmax_compare.cpp.in").read_text()
    text = replace_once(text, "#include <algorithm>", "#include <algorithm>\n#include <immintrin.h>")
    text = replace_once(text, "@@ORIGINAL_ARGMAX@@", original_argmax(header))
    text = replace_once(text, "@@LANES8_ARGMAX@@", LANES8 + ARGMAX)
    text = replace_once(text, "int main() {",
        '__attribute__((noinline)) int32_t avx512(std::span<const float> values) {\n'
        '  if (!Generators::CpuArgMaxAvx512Available()) throw std::runtime_error("AVX512F unavailable");\n'
        '  return Generators::ArgMaxCpuAvx512(values);\n}\n\nint main() {')
    text = replace_once(text, "const auto candidate = lanes8(values);",
        "const auto candidate = avx512(values);\n"
        '      if (lanes8(values) != expected) throw std::runtime_error("Lanes8 control mismatch");')
    text = replace_once(text, "{original, lanes8}", "{original, lanes8, avx512}")
    text = replace_once(text, "31, 32, 33, 65, 257", "31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 257")
    text = replace_once(text, "measure(original);\n    measure(lanes8);",
                        "measure(lanes8);\n    measure(avx512);")
    text = replace_once(text, "const double before = measure(original);", "const double before = measure(lanes8);")
    text = replace_once(text, "const double candidate = measure(lanes8);", "const double candidate = measure(avx512);")
    text = replace_once(text, "const double after = measure(original);", "const double after = measure(lanes8);")
    return text.replace("original_before_s=", "lanes8_before_s=").replace(
        " lanes8_s=", " avx512_s=").replace("original_after_s=", "lanes8_after_s=")


if __name__ == "__main__":
    preflight_path = PARENT / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    parent_manifest = PARENT / "build-manifest.json"
    parent = json.loads(parent_manifest.read_text())
    for name, digest in parent["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    source, header = SOURCE / "src/search.cpp", SOURCE / "src/search.h"
    original_runtime = LEGACY / "libonnxruntime-genai.so"
    assert sha(source) == preflight["source_sha256"] and sha(header) == preflight["header_sha256"]
    assert sha(original_runtime) == preflight["runtime_sha256"]
    assert sha(PARENT / "libonnxruntime-genai.so") == parent["runtime_sha256"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    baseline_object = OUTPUT / "search.baseline.cpp.o"
    subprocess.run([*preflight["compile_command"], "-c", str(source), "-o", str(baseline_object)],
                   cwd=BUILD, check=True)
    assert sha(baseline_object) == sha(PARENT / "search.baseline.cpp.o")
    link = preflight["link_command"].copy()
    object_index = link.index(str(PARENT / "search.baseline.cpp.o"))
    link[object_index] = str(baseline_object)
    link[link.index("-o") + 1] = str(OUTPUT / "baseline-relink.so")
    subprocess.run(link, cwd=BUILD, check=True)
    assert sha(OUTPUT / "baseline-relink.so") == sha(original_runtime)
    target = OUTPUT / "search.cpp"
    target.write_text(transform_source(source.read_text()))
    obj = OUTPUT / "search.cpp.o"
    command = [*preflight["compile_command"], "-iquote", str(source.parent),
               f"-fmacro-prefix-map={target}={source}", "-c", str(target), "-o", str(obj)]
    subprocess.run(command, cwd=BUILD, check=True)
    link[object_index] = str(obj)
    runtime = OUTPUT / "libonnxruntime-genai.so"
    link[link.index("-o") + 1] = str(runtime)
    subprocess.run([*link, "-Wl,--no-undefined"], cwd=BUILD, check=True)
    assert FLAG.encode() in runtime.read_bytes()
    comparator = OUTPUT / "cpu_greedy_argmax_compare.cpp"
    comparator.write_text(comparator_source(header.read_text()))
    executable = OUTPUT / "cpu_greedy_argmax_compare"
    subprocess.run(["c++", "-std=c++20", "-O3", str(comparator), "-o", str(executable)], check=True)
    evidence = [Path(__file__), ROOT / "cpu_greedy_argmax_compare.cpp.in", source, header,
                target, comparator, executable, preflight_path, parent_manifest,
                PARENT / "libonnxruntime-genai.so", original_runtime, baseline_object,
                LEGACY / "indexer_cache.cpp.o", SOURCE / "src/models/env_utils.h"]
    report = dict(status="built_not_qualified", flag=FLAG, default_enabled=False,
        process_static_selector=True, runtime_sha256=sha(runtime),
        parent_runtime_sha256=parent["runtime_sha256"], original_runtime_sha256=sha(original_runtime),
        baseline_relink_byte_identical=True, baseline_search_rebuild_byte_identical=True,
        public_object_layouts_unchanged=True, model_provider_and_sampling_unchanged=True,
        lanes8_control_preserved=True, cpu_feature_gate="avx512f", compile_command=command,
        link_command=link, source_sha256={str(path): sha(path) for path in evidence})
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], runtime_sha256=report["runtime_sha256"])), flush=True)
