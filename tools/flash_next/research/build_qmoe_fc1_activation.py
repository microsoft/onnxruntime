import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/workspace")
PARENT = REPOS / "qwen38-int4-generic-matmul-replay/build-v1"
FROZEN = REPOS / "qwen38-int4-qmoe-node-replay/build-v5"
OUTPUT = REPOS / "qwen38-int4-qmoe-fc1-activation/build-v3"
FLAG = "ORT_WEBGPU_QMOE_FC1_ACTIVATION"
VALIDATE = f'''
  const int fc1_activation = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_activation == 0 || fc1_activation == 1, "{FLAG} must be 0 or 1");
'''
ACTIVATED = '''    TensorShape fc1_activated_shape({static_cast<int64_t>(routed_rows), moe_params.inter_size});
    Tensor& fc1_activated = temporary(3, dtype, fc1_activated_shape);
'''
FUSED = '''
    if (CanFuseFc1Activation(context,
          ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC1_ACTIVATION", 0) == 1)) {
      QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr};
      program.AddInput({hidden_state, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({fc1_experts_weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({fc1_scales, ProgramTensorMetadataDependency::TypeAndRank});
      if (fc1_experts_bias_optional) {
        program.AddInput({fc1_experts_bias_optional, ProgramTensorMetadataDependency::None});
      }
      const auto cols = onnxruntime::narrow<uint32_t>(moe_params.inter_size);
      const auto tiles = CeilDiv(cols * 2u, 8u);
      program.AddInput({&indirect_experts, ProgramTensorMetadataDependency::None})
          .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)
          .AddUniformVariables({hidden_size, cols, tiles, activation_alpha_, activation_beta_, swiglu_limit_})
          .CacheHint(fc1_experts_bias_optional != nullptr);
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
    } else {
'''


def transform(source):
    source = replace_once(source, "Status QMoE::ComputeInternal(ComputeContext& context) const {",
                          '#include "qmoe_fc1_activation.inc"\n\n'
                          "Status QMoE::ComputeInternal(ComputeContext& context) const {" + VALIDATE)
    source = replace_once(source, "key.push_back(context.EnableMatmulFp32Accumulation());",
                          "key.push_back(context.EnableMatmulFp32Accumulation());\n"
                          "  key.push_back(fc1_activation);")
    source = replace_once(source, '  ORT_RETURN_IF_NOT(entry.commands.size() == 5,\n'
                          '                    "QMoE node capture expected exactly five commands, got ", entry.commands.size());',
                          '  const size_t expected_commands = CanFuseFc1Activation(context, fc1_activation == 1) ? 4 : 5;\n'
                          '  ORT_RETURN_IF_NOT(entry.commands.size() == expected_commands,\n'
                          '                    "QMoE node capture command count mismatch, got ", entry.commands.size());')
    start = source.index("    status = is_nvfp4_ || use_int4_cooperative", source.index("if (use_indirect_experts)"))
    end = source.index("    TensorShape fc2_output_shape", start)
    old = source[start:end]
    fallback = replace_once(old, ACTIVATED, "")
    return source[:start] + ACTIVATED + FUSED + fallback + "    }\n\n" + source[end:]


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    frozen = json.loads((FROZEN / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay/contrib_ops/webgpu/moe"
    overlay.mkdir(parents=True)
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = overlay / "qmoe.cc"
    original = FROZEN / "overlay/contrib_ops/webgpu/moe/qmoe.cc"
    source.write_text(transform(original.read_text()))
    header = overlay / "qmoe.h"
    header.write_text(replace_once((original.parent / "qmoe.h").read_text(), "  struct ReplayEntry {",
        "  bool CanFuseFc1Activation(ComputeContext& context, bool enabled) const;\n"
        "  struct ReplayEntry {"))
    shutil.copy2(ROOT / "qmoe_fc1_activation.inc", overlay / "qmoe_fc1_activation.inc")
    command = frozen["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(OUTPUT / "overlay")]
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(FROZEN / "qmoe.cc.o")) == 1
    link[link.index(str(FROZEN / "qmoe.cc.o"))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    text = (ROOT / "qmoe_node_replay_compare.cpp").read_text().replace(
        '"ORT_WEBGPU_QMOE_COMMAND_REPLAY"', f'"{FLAG}"')
    text = replace_once(text,
        'set_flag("ORT_WEBGPU_MATMUL_NBITS_SKINNY", (round / 4) % 2 ? "1" : "0");',
        'set_flag("ORT_WEBGPU_MATMUL_NBITS_SKINNY", "0");')
    comparator.write_text(text)
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, baseline_provider_sha256=parent["provider_sha256"],
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in (source, header, comparator,
            overlay / "qmoe_fc1_activation.inc")},
        original_source_sha256=sha(original), original_header_sha256=sha(original.parent / "qmoe.h"),
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
