import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/workspace")
SOURCE_DIR = REPOS / "onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert"
OLD_MAP = REPOS / "onnxruntime-webgpu-h200-perf/build-int4-qsa-dense-map"
PARENT = REPOS / "qwen38-int4-hc-up-premix/build-v2"
OUTPUT = REPOS / "qwen38-int4-qsa-parallel-map/build-v1"

SELECTOR = """  const int parallel_map = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_PARALLEL_MAP", 0);
  ORT_RETURN_IF_NOT(parallel_map == 0 || parallel_map == 1, "ORT_WEBGPU_QSA_PARALLEL_MAP must be 0 or 1");
  const bool use_parallel_map =
      parallel_map == 1 && use_dense_map && sequence_length == 1 && total_length >= 8192;
  if (use_parallel_map) {
    const uint32_t chunks = DivideRoundUp(ToUint32(total_length), 512u);
    auto counts = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{chunks});
    for (bool count_only : {true, false}) {
      QsaParallelMapProgram map{mask != nullptr, count_only};
      map.CacheHint(mask != nullptr);
      if (mask != nullptr) {
        map.AddInput({mask, ProgramTensorMetadataDependency::Type, {mask->Shape().Size()}, 1});
      }
      if (!count_only) map.AddInput({&counts, ProgramTensorMetadataDependency::Type});
      map.AddOutput({count_only ? &counts : &visible_cache, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{ToUint32(total_length)}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(chunks);
      ORT_RETURN_IF_ERROR(context.RunProgram(map));
    }
  }
"""


def transform(source):
    source = replace_once(source, "Status SparseAttentionIndexer::ComputeQsa(",
                          '#include "qsa_parallel_map.inc"\n\nStatus SparseAttentionIndexer::ComputeQsa(')
    anchor = "  for (auto phase : {QsaCachePhase::Map, QsaCachePhase::Score, QsaCachePhase::Select}) {\n"
    source = replace_once(source, anchor, SELECTOR + anchor +
                          "    if (use_parallel_map && phase == QsaCachePhase::Map) continue;\n")
    return source


def main():
    old = json.loads((OLD_MAP / "build-manifest.json").read_text())
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    for name, digest in old["sources"].items():
        assert sha(SOURCE_DIR / name) == digest, name
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = OUTPUT / "sparse_attention_indexer.cc"
    source.write_text(transform((SOURCE_DIR / source.name).read_text()))
    include = OUTPUT / "qsa_parallel_map.inc"
    shutil.copy2(ROOT / include.name, include)
    obj = OUTPUT / (source.name + ".o")
    compile_command = old["compile_command"].copy()
    compile_command[compile_command.index("-c") + 1] = str(source)
    compile_command[compile_command.index("-o") + 1] = str(obj)
    subprocess.run(compile_command, cwd=parent["working_directory"], env=environment, check=True)
    matches = [p for p in link if p.endswith("contrib_ops__webgpu__bert__sparse_attention_indexer.cc.o")]
    assert len(matches) == 1 and Path(matches[0]).parent == OLD_MAP
    link[link.index(matches[0])] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((ROOT / "qsa_cache_compare.cpp").read_text(),
                                      '"ORT_WEBGPU_QSA_PARALLEL_RANK"',
                                      '"ORT_WEBGPU_QSA_PARALLEL_MAP"'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, baseline_provider_sha256=parent["provider_sha256"],
        compile_commands=[compile_command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, include, comparator)},
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
