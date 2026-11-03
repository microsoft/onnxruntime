import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-fc2-four-columns/build-v2")
FLAG = "ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS"


def transform_program(source):
    source = replace_once(source,
        "explicit QmoeFC2FinalMixProgram(bool bias, bool bitcast)",
        "explicit QmoeFC2FinalMixProgram(bool bias, bool bitcast, bool four_columns)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC2FinalMixBitcastF16"',
        'Program{four_columns ? (bitcast ? "QmoeFC2FinalMixFourColumnsBitcastF16" : "QmoeFC2FinalMixFourColumnsF16") : bitcast ? "QmoeFC2FinalMixBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
        "bias_{bias}, bitcast_{bitcast}, columns_{four_columns ? 4u : 2u} {}")
    source = replace_once(source,
        '<< "var<workgroup> partials: array<array<f16, 32>, 20>;\\n";',
        '<< "var<workgroup> partials: array<array<f16, 32>, " << 10u * columns_ << ">;\\n";')
    for original, replacement in (
            ("let base_col = workgroup_idx * 2u;", 'let base_col = workgroup_idx * " << columns_ << "u;'),
            ("for (var c = 0u; c < 2u; c++)", 'for (var c = 0u; c < " << columns_ << "u; c++)'),
            ("partials[rank * 2u + c]", 'partials[rank * " << columns_ << "u + c]'),
            ("if (local_idx < 20u)", 'if (local_idx < " << 10u * columns_ << "u)'),
            ("let r = local_idx / 2u;", 'let r = local_idx / " << columns_ << "u;'),
            ("let c = local_idx % 2u;", 'let c = local_idx % " << columns_ << "u;'),
            ("if (local_idx < 2u)", 'if (local_idx < " << columns_ << "u)'),
            ("partials[r * 2u + local_idx]", 'partials[r * " << columns_ << "u + local_idx]')):
        source = replace_once(source, original, replacement)
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  uint32_t columns_;")


def transform_qmoe(source):
    original = '  ORT_RETURN_IF_NOT(expert_bitcast == 0 || expert_bitcast == 1, "ORT_WEBGPU_EXPERT_BITCAST_UNPACK must be 0 or 1");'
    assert source.count(original) == 2
    validation = original + f'''
  const int fc2_four_columns = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc2_four_columns == 0 || fc2_four_columns == 1, "{FLAG} must be 0 or 1");'''
    source = source.replace(original, validation)
    source = replace_once(source, "  key.push_back(expert_bitcast);",
        "  key.push_back(expert_bitcast);\n  key.push_back(fc2_four_columns);")
    source = replace_once(source,
        "      QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1};",
        """      const bool four_columns = fc2_four_columns == 1 &&
          context.DeviceLimits().maxComputeWorkgroupStorageSize >= 15360;
      QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1, four_columns};""")
    return replace_once(source, ".SetWorkgroupSize(320).SetDispatchGroupSize(1280)",
        ".SetWorkgroupSize(320).SetDispatchGroupSize(four_columns ? 640 : 1280)")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay/contrib_ops/webgpu/moe"
    shutil.copytree(COMPONENT / "overlay/contrib_ops/webgpu/moe", overlay)
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = overlay / "qmoe.cc"
    source.write_text(transform_qmoe(source.read_text()))
    program = overlay / "qmoe_fc2_final_mix.inc"
    program.write_text(transform_program(program.read_text()))
    command = component["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(OUTPUT / "overlay")]
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(COMPONENT / obj.name)) == 1
    link[link.index(str(COMPONENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    text = (COMPONENT / comparator.name).read_text()
    assert text.count('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"') == 4
    comparator.write_text(text.replace('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"', f'"{FLAG}"'))
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
        fc2_columns=[2, 4], four_column_workgroup_storage_bytes=15360,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
