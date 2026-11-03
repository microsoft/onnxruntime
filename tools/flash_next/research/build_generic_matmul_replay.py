import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/workspace")
FROZEN = REPOS / "qwen38-int4-static-node-replay/build-v8"
PARENT = REPOS / "qwen38-int4-hc-up-premix/build-v2"
OUTPUT = REPOS / "qwen38-int4-generic-matmul-replay/build-v1"
SELECTOR = """
  const int generic_replay =
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_GENERIC_MATMUL_REPLAY", 0);
  ORT_RETURN_IF_NOT(generic_replay == 0 || generic_replay == 1,
                    "ORT_WEBGPU_GENERIC_MATMUL_REPLAY must be 0 or 1");
  const bool generic_shape =
      (K_ == 2560 && (N_ == 1 || N_ == 16384 || N_ == 7168 || N_ == 6144 ||
                     N_ == 10240 || N_ == 2560)) ||
      ((K_ == 6144 || K_ == 640) && N_ == 2560) ||
      (K_ == 10240 && N_ == 320);
  const bool generic_route =
      generic_replay == 1 && generic_shape &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      bits_ == 4 && block_size_ == 32 && !has_zero_points && bias == nullptr &&
      !context.EnableMatmulFp32Accumulation() &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) == 0 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", 0) == 0;
  if (!exact_route && !generic_route) return compute();
"""


def transform(source):
    return replace_once(source, "  if (!exact_route) return compute();", SELECTOR)


def main():
    frozen = json.loads((FROZEN / "build-manifest.json").read_text())
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    old_source = FROZEN / "overlay/contrib_ops/webgpu/quantization/matmul_nbits.cc"
    assert sha(old_source) == frozen["overlay_source_sha256"]["contrib_ops/webgpu/quantization/matmul_nbits.cc"]
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
    source = OUTPUT / "matmul_nbits.cc"
    source.write_text(transform(old_source.read_text()))
    obj = OUTPUT / (source.name + ".o")
    command = next(c.copy() for c in frozen["compile_commands"]
                   if c[c.index("-c") + 1] == str(old_source))
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(FROZEN / "matmul_nbits.cc.o")) == 1
    link[link.index(str(FROZEN / "matmul_nbits.cc.o"))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((ROOT / "static_node_replay_compare.cpp").read_text(),
        'setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1)',
        'setenv("ORT_WEBGPU_GENERIC_MATMUL_REPLAY", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, baseline_provider_sha256=parent["provider_sha256"],
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, comparator)},
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
