import json
import os
from pathlib import Path
import subprocess

from build_dense_bitcast_unpack import transform_replay
from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-hc-up-premix/build-v2")
FROZEN = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
OUTPUT = Path("/workspace/qwen38-int4-hc-up-tiled/build-v1")
FLAG = "ORT_WEBGPU_HC_UP_TILED"


def tiled_program(source):
    start = source.index("class HCUpPreMixProgram")
    end = source.index("class HCUpPreMixF16", start)
    result = source[start:end].replace("HCUpPreMixProgram", "HCUpPreMixTiledProgram")
    result = replace_once(result, 'Program{"HCUpPreMixF16"}', 'Program{"HCUpPreMixTiledF16"}')
    for original, replacement in (
            ("partials: array<f16, 128>", "partials: array<f16, 1024>"),
            ("gates: array<f16, 4>", "gates: array<f16, 32>"),
            ('<< "let col = workgroup_idx + branch * 2560u;\\n"',
             '<< "let base_col = workgroup_idx * 8u;\\n"\n'
             '         << "for (var offset = 0u; offset < 8u; offset++) {\\n"\n'
             '         << "let col = base_col + offset + branch * 2560u;\\n"'),
            ('<< "partials[local_idx] = total;\\n"',
             '<< "partials[offset * 128u + local_idx] = total;\\n"\n'
             '         << "}\\n"'),
            ('<< "if (lane == 0u) {\\n"',
             '<< "if (local_idx < 32u) {\\n"\n'
             '         << "  let offset = local_idx / 4u;\\n"\n'
             '         << "  let c = local_idx % 4u;\\n"'),
            ("value += partials[local_idx + i]", "value += partials[offset * 128u + c * 32u + i]"),
            ("gates[branch] =", "gates[local_idx] ="),
            ("if (local_idx == 0u)", "if (local_idx < 8u)"),
            ('streams.GetByOffset("c * 2560u + workgroup_idx")',
             'streams.GetByOffset("c * 2560u + base_col + local_idx")'),
            ("f32(gates[c])", "f32(gates[local_idx * 4u + c])"),
            ('y.SetByOffset("workgroup_idx",', 'y.SetByOffset("base_col + local_idx",')):
        result = replace_once(result, original, replacement)
    return result


def transform(source):
    source = replace_once(source, "class HCUpPreMixF16", tiled_program(source) + "class HCUpPreMixF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int tiled = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(tiled == 0 || tiled == 1, "{FLAG} must be 0 or 1");''')
    start = source.index("      return static_replay_.Run(context, {y}, [&]() -> Status {")
    end = source.index("\n    }\n    auto projected", start)
    block = source[start:end]
    setup = block[block.index("        program.SetWorkgroupSize"):block.index("      });")]
    setup = replace_once(setup, ".SetDispatchGroupSize(2560)", ".SetDispatchGroupSize(tiled == 1 ? 320 : 2560)")
    replacement = (
        "      auto run = [&]<typename P>(P& program) -> Status {\n" + setup +
        "      };\n"
        "      return static_replay_.Run(context, {y}, [&]() -> Status {\n"
        "        if (tiled == 1) {\n"
        "          HCUpPreMixTiledProgram program;\n"
        "          return run(program);\n"
        "        }\n"
        "        HCUpPreMixProgram program;\n"
        "        return run(program);\n"
        "      }, 0, tiled);")
    return source[:start] + replacement + source[end:]


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay"
    overlay.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    (overlay / "static_node_replay.h").write_text(
        transform_replay((FROZEN / "overlay/static_node_replay.h").read_text()))
    source = OUTPUT / "hc_up_premix.cc"
    source.write_text(transform((COMPONENT / source.name).read_text()))
    obj = OUTPUT / (source.name + ".o")
    command = next(c.copy() for c in component["compile_commands"]
                   if c[c.index("-c") + 1] == str(COMPONENT / source.name))
    command[1:1] = ["-I" + str(overlay)]
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(COMPONENT / obj.name)) == 1
    link[link.index(str(COMPONENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((COMPONENT / comparator.name).read_text(),
        'setenv("ORT_WEBGPU_HC_UP_PREMIX", value, 1)', f'setenv("{FLAG}", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in (source, comparator, overlay / "static_node_replay.h")},
        comparator_sha256=sha(OUTPUT / "native_compare"), no_existing_object_layout_changes=True,
        columns_per_workgroup=[1, 8], workgroups=[2560, 320], replay_variant_keyed=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
