import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-generic-matmul-replay/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-small-m-prefill/build-v4")
FLAG = "ORT_WEBGPU_SMALL_M_PREFILL"


def transform(source):
    source = replace_once(source, "namespace {\n", 'namespace {\n#include "small_m_prefill.inc"\n')
    validation = f'''
  const int small_m_prefill = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(small_m_prefill == 0 || small_m_prefill == 1, "{FLAG} must be 0 or 1");
'''
    start = source.index("Status ApplyMatMulNBits(")
    body = source.index("{", start) + 1
    source = source[:body] + validation + source[body:]
    source = replace_once(source,
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {",
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {" + validation)
    return replace_once(source, "    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K,",
        """    if (small_m_prefill == 1 && M == 16 && N >= 16 && K % 32 == 0 && dispatch_M == M &&
        nbits == 4 && block_size == 32 && batch_count == 1 &&
        !has_zero_points && !has_bias && !has_weight_idx &&
        a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
        subgroup_matrix_config->Is(16, 16, 16) && subgroup_matrix_config->subgroupSize == 32 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
        context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
      return ApplySmallMPrefill(a, b, scales, N, K, *subgroup_matrix_config, context, y);
    }
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K,""")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    original = COMPONENT / "matmul_nbits.cc"
    assert sha(original) == component["source_sha256"][original.name]
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
    source.write_text(transform(original.read_text()))
    shutil.copy2(ROOT / "small_m_prefill.inc", OUTPUT)
    obj = OUTPUT / (source.name + ".o")
    command = component["compile_commands"][0].copy()
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(COMPONENT / obj.name)) == 1
    link[link.index(str(COMPONENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    native = replace_once((ROOT / "static_node_replay_compare.cpp").read_text(),
        'setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1)', f'setenv("{FLAG}", value, 1)')
    native = replace_once(native, "      options.SetIntraOpNumThreads(1);",
        f'''      options.SetIntraOpNumThreads(1);
      const auto profile_prefix = std::string("/evidence/webgpu-perf-oct20/small-m-prefill-{OUTPUT.name}-native-") +
          std::filesystem::path(argv[file]).stem().string();
      options.EnableProfiling(profile_prefix.c_str());''')
    native = replace_once(native, '      flag("1");\n      std::cout << "PASS "',
        '''      flag("1");
      auto profile_path = session.EndProfilingAllocated(allocator);
      std::cout << "PROFILE " << profile_path.get() << std::endl;
      std::cout << "PASS "''')
    comparator.write_text(native)
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, comparator, OUTPUT / "small_m_prefill.inc")},
        comparator_sha256=sha(OUTPUT / "native_compare"), no_existing_object_layout_changes=True,
        only_symmetric_fp16_int4_m16=True, original_subgroup_config_selection_preserved=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
