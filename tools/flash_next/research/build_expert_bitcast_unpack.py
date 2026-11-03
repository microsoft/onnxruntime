import json
import os
from pathlib import Path
import shutil
import subprocess

from build_mlp_bitcast_unpack import DECODE_IMPLEMENTATION
from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-mlp-input-projections/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-qmoe-hierarchical-gate/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
FLAG = "ORT_WEBGPU_EXPERT_BITCAST_UNPACK"


def transform_program(source, kind):
    name = "QmoeFC1SwiGLU" if kind == "fc1" else "QmoeFC2FinalMix"
    source = replace_once(source,
        f'explicit {name}Program(bool bias) : Program{{"{name}F16"}}, bias_{{bias}} {{}}',
        f'explicit {name}Program(bool bias, bool bitcast) : '
        f'Program{{bitcast ? "{name}BitcastF16" : "{name}F16"}}, bias_{{bias}}, bitcast_{{bitcast}} {{}}')
    source = replace_once(source, "    auto& code = shader.MainFunctionBody();", DECODE_IMPLEMENTATION)
    indent = "        " if kind == "fc1" else "      "
    for name, argument, original in (
            ("low", "packed[i]", "packed[i] & 0x0f0f0f0fu"),
            ("high", "packed[i] >> 4u", "(packed[i] >> 4u) & 0x0f0f0f0fu")):
        literal = f'"{indent}let {name} = vec4<f16>(unpack4xU8({original})) - vec4<f16>(8);\\n"'
        source = replace_once(source, literal,
            f'(bitcast_ ? "{indent}let {name} = decode_low_nibbles({argument});\\n" : ' + literal + ")")
    return replace_once(source, "  bool bias_;", "  bool bias_;\n  bool bitcast_;")


def transform_qmoe(source):
    validation = f'''
  const int expert_bitcast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(expert_bitcast == 0 || expert_bitcast == 1, "{FLAG} must be 0 or 1");
'''
    source = replace_once(source, "Status QMoE::ComputeInternal(ComputeContext& context) const {",
        "Status QMoE::ComputeInternal(ComputeContext& context) const {" + validation)
    source = replace_once(source, "  const Tensor* hidden_state = context.Input<Tensor>(0);",
        validation + "  const Tensor* hidden_state = context.Input<Tensor>(0);")
    source = replace_once(source, "  key.push_back(hierarchical_gate);",
        "  key.push_back(hierarchical_gate);\n  key.push_back(expert_bitcast);")
    for name, bias in (("QmoeFC1SwiGLU", "fc1"), ("QmoeFC2FinalMix", "fc2")):
        source = replace_once(source,
            f"{name}Program program{{{bias}_experts_bias_optional != nullptr}};",
            f"{name}Program program{{{bias}_experts_bias_optional != nullptr, expert_bitcast == 1}};")
    return source


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
    for kind, filename in (("fc1", "qmoe_fc1_activation.inc"), ("fc2", "qmoe_fc2_final_mix.inc")):
        path = overlay / filename
        path.write_text(transform_program(path.read_text(), kind))
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
    assert text.count('"ORT_WEBGPU_QMOE_HIERARCHICAL_GATE"') == 4
    comparator.write_text(text.replace('"ORT_WEBGPU_QMOE_HIERARCHICAL_GATE"', f'"{FLAG}"'))
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
        comparator_sha256=sha(OUTPUT / "native_compare"), no_object_layout_changes=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
