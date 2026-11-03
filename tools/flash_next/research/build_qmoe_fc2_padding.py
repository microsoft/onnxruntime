import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-qmoe-fc2-final-mix/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-fc2-padding/build-v1")
FLAG = "ORT_WEBGPU_QMOE_FC2_PADDED_PARTIALS"


def transform(source):
    source = replace_once(source,
        "Status QMoE::ComputeInternal(ComputeContext& context) const {",
        f'''Status QMoE::ComputeInternal(ComputeContext& context) const {{
  const int fc2_padding = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc2_padding == 0 || fc2_padding == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  key.push_back(fc2_mix);",
        "  key.push_back(fc2_mix);\n  key.push_back(fc2_padding);")
    source = replace_once(source,
        "      QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr};",
        f'''      const bool padded = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0) == 1 &&
          context.DeviceLimits().maxComputeWorkgroupStorageSize >= 14120;
      QmoeFC2FinalMixProgram program{{fc2_experts_bias_optional != nullptr, padded}};''')
    return replace_once(source, ".CacheHint(fc2_experts_bias_optional != nullptr);",
        ".CacheHint(fc2_experts_bias_optional != nullptr, padded);")


def transform_shader(source):
    source = replace_once(source,
        'explicit QmoeFC2FinalMixProgram(bool bias) : Program{"QmoeFC2FinalMixF16"}, bias_{bias} {}',
        '''QmoeFC2FinalMixProgram(bool bias, bool padded)
      : Program{padded ? "QmoeFC2FinalMixF16Padded" : "QmoeFC2FinalMixF16"},
        bias_{bias}, padded_{padded} {}''')
    source = replace_once(source,
        '<< "var<workgroup> partials: array<array<f16, 32>, 20>;\\n";',
        '<< "var<workgroup> partials: array<array<f16, " << (padded_ ? 33 : 32) << ">, 20>;\\n";')
    return replace_once(source, "  bool bias_;", "  bool bias_;\n  bool padded_;")


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
    shader = overlay / "qmoe_fc2_final_mix.inc"
    source.write_text(transform(source.read_text()))
    shader.write_text(transform_shader(shader.read_text()))
    command = parent["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(OUTPUT / "overlay")]
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(PARENT / "qmoe.cc.o")) == 1
    link[link.index(str(PARENT / "qmoe.cc.o"))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator_source = (PARENT / "native_compare.cpp").read_text()
    assert comparator_source.count('"ORT_WEBGPU_QMOE_FC2_FINAL_MIX"') == 4
    comparator.write_text(comparator_source.replace('"ORT_WEBGPU_QMOE_FC2_FINAL_MIX"', f'"{FLAG}"'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, baseline_provider_sha256=parent["provider_sha256"],
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in (*sorted(overlay.iterdir()), comparator)
                       if p.is_file()},
        comparator_sha256=sha(OUTPUT / "native_compare"),
        no_object_layout_changes=True, quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
