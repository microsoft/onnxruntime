import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-qmoe-fusions-map/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-qmoe-fc2-final-mix/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-activation-padding/build-v1")
FLAG = "ORT_WEBGPU_QMOE_PADDED_ACTIVATIONS"


def transform(source):
    source = replace_once(source,
        "Status QMoE::ComputeInternal(ComputeContext& context) const {",
        f'''Status QMoE::ComputeInternal(ComputeContext& context) const {{
  const int activation_padding = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(activation_padding == 0 || activation_padding == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  key.push_back(fc2_mix);",
        "  key.push_back(fc2_mix);\n  key.push_back(activation_padding);")
    source = replace_once(source, "std::array<std::optional<Tensor>, 5>* scratch) const {",
        f'''std::array<std::optional<Tensor>, 5>* scratch) const {{
  const bool padded_activations = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0) == 1 &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 15680;''')
    for number, name in ((1, "QmoeFC1SwiGLUProgram"), (2, "QmoeFC2FinalMixProgram")):
        bias = f"fc{number}_experts_bias_optional != nullptr"
        source = replace_once(source, f"{name} program{{{bias}}};",
            f"{name} program{{{bias}, padded_activations}};")
        source = replace_once(source, f".CacheHint({bias});",
            f".CacheHint({bias}, padded_activations);")
    return source


def transform_shader(source, number):
    name = "QmoeFC1SwiGLU" if number == 1 else "QmoeFC2FinalMix"
    source = replace_once(source,
        f'explicit {name}Program(bool bias) : Program{{"{name}F16"}}, bias_{{bias}} {{}}',
        f'''{name}Program(bool bias, bool padded)
      : Program{{padded ? "{name}ActivationPaddedF16" : "{name}F16"}},
        bias_{{bias}}, padded_{{padded}} {{}}''')
    source = replace_once(source, "  bool bias_;", "  bool bias_;\n  bool padded_;")
    old, new = (256, 288) if number == 1 else (1600, 1800)
    source = replace_once(source,
        f'<< "var<workgroup> tile_a: array<vec4<f16>, {old}>;\\n"',
        f'<< "var<workgroup> tile_a: array<vec4<f16>, " << (padded_ ? {new} : {old}) << ">;\\n"')
    expected_stores = 2 if number == 1 else 1
    assert source.count("tile_a[id]") == expected_stores
    source = source.replace("tile_a[id]",
        'tile_a[" << (padded_ ? "(id / 8u) * 9u + id % 8u" : "id") << "]')
    old_index = "lane * 8u + " if number == 1 else "rank * 160u + lane * 8u + "
    new_index = ('lane * " << (padded_ ? 9 : 8) << "u + ' if number == 1 else
                 'rank * " << (padded_ ? 180 : 160) << "u + lane * " << (padded_ ? 9 : 8) << "u + ')
    assert source.count(old_index) == 2
    return source.replace(old_index, new_index)


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
    source.write_text(transform(source.read_text()))
    for number, name in ((1, "qmoe_fc1_activation.inc"), (2, "qmoe_fc2_final_mix.inc")):
        shader = overlay / name
        shader.write_text(transform_shader(shader.read_text(), number))
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
    comparator_source = (COMPONENT / comparator.name).read_text()
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
        comparator_sha256=sha(OUTPUT / "native_compare"), no_object_layout_changes=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
