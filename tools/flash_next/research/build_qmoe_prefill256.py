import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-qsa-indexer-stack/build-v1")
FROZEN = Path("/workspace/qwen38-int4-qmoe-extended-prefill/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1")
FLAG = "ORT_WEBGPU_QMOE_PREFILL256"
ROWS = (16, 128, 129, 255, 256, 257)


def transform(source):
    validation = ('  ORT_RETURN_IF_NOT(extended_prefill == 0 || extended_prefill == 1, '
                  '"ORT_WEBGPU_QMOE_EXTENDED_PREFILL must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int prefill256 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill256 == 0 || prefill256 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(extended_prefill);",
                          "  key.push_back(extended_prefill);\n  key.push_back(prefill256);")
    source = replace_once(source, "  const int64_t max_indirect_tokens =",
        '''  const bool use_prefill256 =
      prefill256 == 1 && use_extended_int4_prefill && moe_params.num_rows == 256 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 12800;
  const int64_t max_indirect_tokens =''')
    source = replace_once(source,
        "      use_nvfp4_prefill || use_extended_int4_prefill ? 128 : use_integer_prefill ? 16 : 8;",
        "      use_prefill256 ? 256 : use_nvfp4_prefill || use_extended_int4_prefill ? 128 : use_integer_prefill ? 16 : 8;")
    return replace_once(source,
        "      use_nvfp4_prefill ? 4096 : use_extended_int4_prefill ? 1280 : use_integer_prefill ? 256 : 128;",
        "      use_prefill256 ? 2560 : use_nvfp4_prefill ? 4096 : use_extended_int4_prefill ? 1280 : use_integer_prefill ? 256 : 128;")


def transform_compare(source):
    source = source.replace("ORT_WEBGPU_QMOE_EXTENDED_PREFILL", FLAG)
    source = replace_once(source, "extended-prefill-v1-native-", "prefill256-v1-native-")
    source = replace_once(source, "{16, 32, 64, 128, 129}", "{16, 128, 129, 255, 256, 257}")
    source = replace_once(source, "rows == 129 ? 2 : 8", "rows > 128 && rows != 256 ? 2 : 8")
    source = replace_once(source, "rows == 129 ? rows : 16", "rows > 128 && rows != 256 ? rows : 16")
    return replace_once(source, "last_router.data(), 129", "last_router.data(), 257")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    frozen = json.loads((FROZEN / "build-manifest.json").read_text())
    assert sha(PARENT / "libonnxruntime_providers_webgpu.so") == parent["provider_sha256"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay/contrib_ops/webgpu/moe"
    shutil.copytree(FROZEN / "overlay/contrib_ops/webgpu/moe", overlay)
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Indexer parent relink mismatch"
    library.unlink()
    source = overlay / "qmoe.cc"
    source.write_text(transform(source.read_text()))
    command = frozen["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(OUTPUT / "overlay")]
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(FROZEN / obj.name)) == 1
    link[link.index(str(FROZEN / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(transform_compare((ROOT / "qmoe_extended_prefill_compare.cpp").read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        comparator_sha256=sha(OUTPUT / "native_compare"),
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in (*sorted(overlay.iterdir()), comparator)
                       if p.is_file()},
        tokens_exact=256, routed_rows_cap=2560, minimum_dispatch_limit=12800,
        reference_arithmetic="retained_chunk16", no_existing_object_layout_changes=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
