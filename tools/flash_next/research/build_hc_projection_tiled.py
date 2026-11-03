import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-qmoe-fusions-map/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-hc-projection-silu/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-hc-projection-tiled/build-v2")
FLAG = "ORT_WEBGPU_HC_PROJECTION_TILED"


def transform(source):
    source = replace_once(source,
        'HCProjectionSiLUProgram() : Program{"HCProjectionSiLUF16"} {}',
        '''explicit HCProjectionSiLUProgram(bool tiled)
      : Program{tiled ? "HCProjectionSiLUTiledF16" : "HCProjectionSiLUF16"}, tiled_{tiled} {}''')
    source = replace_once(source,
        '    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, 64>;\\n";',
        '''    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, "
        << (tiled_ ? 256 : 64) << ">;\\n";
    if (tiled_) shader.AdditionalImplementation()
        << "var<workgroup> tile_a: array<vec4<f16>, 2880>;\\n";''')
    source = replace_once(source, '    code << "let lane = local_idx % 32u;\\n"',
        '''    if (tiled_) {
      code << "for (var id = local_idx; id < 2560u; id += 256u) {\\n"
           << "tile_a[(id / 8u) * 9u + id % 8u] = " << a.GetByOffset("id") << ";\\n"
           << "}\\nworkgroupBarrier();\\n";
    }
    code << "let lane = local_idx % 32u;\\n"''')
    source = replace_once(source,
        '<< "let col = workgroup_idx * 2u + local_idx / 32u;\\n"',
        '<< "let col = workgroup_idx * " << (tiled_ ? 8 : 2) << "u + local_idx / 32u;\\n"')
    source = replace_once(source,
        '<< "var total = f16(0);\\n"\n         << "for (var tile',
        '''<< "var total = f16(0);\\n";
    if (tiled_) code << "if (col < 324u) {\\n";
    code << "for (var tile''')
    for offset in ("block * 8u + i * 2u", "block * 8u + i * 2u + 1u"):
        source = replace_once(source, f'a.GetByOffset("{offset}")',
            f'(tiled_ ? "tile_a[{offset.replace("block * 8u", "block * 9u")}]" : a.GetByOffset("{offset}"))')
    source = replace_once(source,
        '<< "}\\n"\n         << "partials[local_idx] = total;\\n"',
        '''<< "}\\n";
    if (tiled_) code << "}\\n";
    code << "partials[local_idx] = total;\\n"''')
    source = replace_once(source, '<< "if (lane == 0u) {\\n"',
        '<< (tiled_ ? "if (lane == 0u && col < 324u) {\\n" : "if (lane == 0u) {\\n")')
    source = replace_once(source,
        '{"alpha", ProgramUniformVariableDataType::Float32});\n};',
        '{"alpha", ProgramUniformVariableDataType::Float32});\n private:\n  bool tiled_;\n};')
    source = replace_once(source,
        '    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_SILU", 0);',
        f'''    const int tiled_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(tiled_mode == 0 || tiled_mode == 1, "{FLAG} must be 0 or 1");
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_SILU", 0);''')
    source = replace_once(source,
        "      return static_replay_.Run(context, {activated, tail},",
        '''      const auto& limits = context.DeviceLimits();
      const bool tiled = tiled_mode == 1 && limits.maxComputeWorkgroupSizeX >= 256 &&
          limits.maxComputeInvocationsPerWorkgroup >= 256 && limits.maxComputeWorkgroupStorageSize >= 23552;
      auto& replay = tiled ? tiled_replay_ : static_replay_;
      return replay.Run(context, {activated, tail},''')
    source = replace_once(source, "        HCProjectionSiLUProgram program;",
        "        HCProjectionSiLUProgram program{tiled};")
    source = replace_once(source, "program.SetWorkgroupSize(64).SetDispatchGroupSize(162);",
        "program.SetWorkgroupSize(tiled ? 256 : 64).SetDispatchGroupSize(tiled ? 41 : 162);")
    return replace_once(source,
        '  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"HCProjectionSiLUF16", {0, 2, 3}};',
        '''  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"HCProjectionSiLUF16", {0, 2, 3}};
  mutable onnxruntime::webgpu::StaticNodeReplay tiled_replay_{"HCProjectionSiLUTiledF16", {0, 2, 3}};''')


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
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
    source = OUTPUT / "hc_projection_silu.cc"
    original = COMPONENT / source.name
    source.write_text(transform(original.read_text()))
    command = component["compile_commands"][0].copy()
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / (source.name + ".o")
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    original_obj = COMPONENT / obj.name
    assert link.count(str(original_obj)) == 1
    link[link.index(str(original_obj))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((COMPONENT / comparator.name).read_text(),
        'setenv("ORT_WEBGPU_HC_PROJECTION_SILU", value, 1)', f'setenv("{FLAG}", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, comparator)},
        original_source_sha256=sha(original), original_object_sha256=sha(original_obj),
        comparator_sha256=sha(OUTPUT / "native_compare"),
        no_runtime_object_layout_changes=True, private_kernel_layout_changed=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
