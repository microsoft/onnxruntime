import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-mlp-input-projections/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-mlp-bitcast-unpack/build-v1")
FLAG = "ORT_WEBGPU_MLP_BITCAST_UNPACK"
DECODE_IMPLEMENTATION = '''    if (bitcast_) {
      shader.AdditionalImplementation() <<
          "fn decode_low_nibbles(word: u32) -> vec4<f16> {\\n"
          "  let bottom = bitcast<vec2<f16>>((word & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);\\n"
          "  let top = bitcast<vec2<f16>>(((word >> 8u) & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);\\n"
          "  return vec4<f16>(bottom.x, top.x, bottom.y, top.y);\\n"
          "}\\n";
    }
    auto& code = shader.MainFunctionBody();'''


def transform(source):
    source = replace_once(source,
        '  MlpInputProjectionsProgram() : Program{"MlpInputProjectionsF16"} {}',
        '''  explicit MlpInputProjectionsProgram(bool bitcast)
      : Program{bitcast ? "MlpInputProjectionsBitcastF16" : "MlpInputProjectionsF16"}, bitcast_{bitcast} {}''')
    source = replace_once(source, "    auto& code = shader.MainFunctionBody();",
        DECODE_IMPLEMENTATION)
    for word, indent in (("packed", "        "), ("up_packed", "          ")):
        for name, argument, original in (
                ("low", f"{word}[i]", f"{word}[i] & 0x0f0f0f0fu"),
                ("high", f"{word}[i] >> 4u", f"({word}[i] >> 4u) & 0x0f0f0f0fu")):
            literal = f'"{indent}let {name} = vec4<f16>(unpack4xU8({original})) - vec4<f16>(8);\\n"'
            replacement = (f'(bitcast_ ? "{indent}let {name} = decode_low_nibbles({argument});\\n" : '
                           + literal + ")")
            source = replace_once(source, literal, replacement)
    source = replace_once(source, "\n};\n\nclass MlpInputProjectionsF16",
        "\n private:\n  bool bitcast_;\n};\n\nclass MlpInputProjectionsF16")
    source = replace_once(source, "    const auto* a = context.Input(0);",
        f'''    const int bitcast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(bitcast == 0 || bitcast == 1, "{FLAG} must be 0 or 1");
    const auto* a = context.Input(0);''')
    source = replace_once(source, "      return replay_.Run(context,",
        "      auto& replay = bitcast == 1 ? bitcast_replay_ : replay_;\n      return replay.Run(context,")
    source = replace_once(source, "        MlpInputProjectionsProgram program;",
        "        MlpInputProjectionsProgram program{bitcast == 1};")
    return replace_once(source,
        '  mutable StaticNodeReplay replay_{"MlpInputProjectionsF16", {0, 1, 2, 3, 4, 5, 6, 7, 8}};',
        '''  mutable StaticNodeReplay replay_{"MlpInputProjectionsF16", {0, 1, 2, 3, 4, 5, 6, 7, 8}};
  mutable StaticNodeReplay bitcast_replay_{"MlpInputProjectionsBitcastF16", {0, 1, 2, 3, 4, 5, 6, 7, 8}};''')


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
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
    source = OUTPUT / "mlp_input_projections.cc"
    source.write_text(transform((PARENT / source.name).read_text()))
    command = parent["compile_commands"][0].copy()
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / (source.name + ".o")
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(PARENT / obj.name)) == 1
    link[link.index(str(PARENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((PARENT / comparator.name).read_text(),
        'setenv("ORT_WEBGPU_MLP_INPUT_PROJECTIONS", value, 1)', f'setenv("{FLAG}", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, comparator)},
        comparator_sha256=sha(OUTPUT / "native_compare"), no_existing_object_layout_changes=True,
        independent_unpack_replay_caches=True, quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
