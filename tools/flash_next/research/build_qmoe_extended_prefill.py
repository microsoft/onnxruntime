import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-extended-prefill/build-v1")
FLAG = "ORT_WEBGPU_QMOE_EXTENDED_PREFILL"


def transform(source):
    validation = '  ORT_RETURN_IF_NOT(expert_bitcast == 0 || expert_bitcast == 1, "ORT_WEBGPU_EXPERT_BITCAST_UNPACK must be 0 or 1");'
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int extended_prefill = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(extended_prefill == 0 || extended_prefill == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(expert_bitcast);",
        "  key.push_back(expert_bitcast);\n  key.push_back(extended_prefill);")
    source = replace_once(source,
        "  const int64_t max_indirect_tokens = use_nvfp4_prefill ? 128 : use_integer_prefill ? 16 : 8;",
        """  const bool use_extended_int4_prefill =
      extended_prefill == 1 && use_int4_prefill && block_size_ == 32 &&
      hidden_state->IsDataType<MLFloat16>() && is_fused_swiglu &&
      moe_params.hidden_size == 2560 && moe_params.inter_size == 640 &&
      num_experts == 512 && k_ == 10 && fc3_experts_weights_optional == nullptr &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      !context.EnableMatmulFp32Accumulation() &&
      device_limits.maxComputeWorkgroupsPerDimension >= 2560;
  const int64_t max_indirect_tokens =
      use_nvfp4_prefill || use_extended_int4_prefill ? 128 : use_integer_prefill ? 16 : 8;""")
    return replace_once(source,
        "  const int64_t max_indirect_rows = use_nvfp4_prefill ? 4096 : use_integer_prefill ? 256 : 128;",
        """  const int64_t max_indirect_rows =
      use_nvfp4_prefill ? 4096 : use_extended_int4_prefill ? 1280 : use_integer_prefill ? 256 : 128;""")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay/contrib_ops/webgpu/moe"
    shutil.copytree(PARENT / "overlay/contrib_ops/webgpu/moe", overlay)
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = overlay / "qmoe.cc"
    source.write_text(transform(source.read_text()))
    command = parent["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(OUTPUT / "overlay")]
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(PARENT / obj.name)) == 1
    link[link.index(str(PARENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    shutil.copy2(ROOT / "qmoe_extended_prefill_compare.cpp", comparator)
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in (*sorted(overlay.iterdir()), comparator)
                       if p.is_file()},
        comparator_sha256=sha(OUTPUT / "native_compare"), no_existing_object_layout_changes=True,
        tokens_cap=128, routed_rows_cap=1280, reference_arithmetic="retained_chunk16",
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
