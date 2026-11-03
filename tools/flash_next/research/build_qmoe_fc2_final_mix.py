import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/workspace")
PARENT = REPOS / "qwen38-int4-qmoe-fc1-activation/build-v3"
OUTPUT = REPOS / "qwen38-int4-qmoe-fc2-final-mix/build-v2"
FLAG = "ORT_WEBGPU_QMOE_FC2_FINAL_MIX"
FUSED = '''
    if (CanFuseFc2FinalMix(context,
          ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", 0) == 1)) {
      QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr};
      program.AddInput({&fc1_activated, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({fc2_experts_weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({fc2_scales, ProgramTensorMetadataDependency::TypeAndRank});
      if (fc2_experts_bias_optional) {
        program.AddInput({fc2_experts_bias_optional, ProgramTensorMetadataDependency::None});
      }
      program.AddInput({&router_values, ProgramTensorMetadataDependency::Type})
          .AddInput({&indirect_experts, ProgramTensorMetadataDependency::Type})
          .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(320).SetDispatchGroupSize(1280)
          .AddUniformVariables({hidden_size, k, num_experts})
          .CacheHint(fc2_experts_bias_optional != nullptr);
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
    } else {
'''


def transform(source):
    source = replace_once(source, '#include "qmoe_fc1_activation.inc"',
        '#include "qmoe_fc1_activation.inc"\n#include "qmoe_fc2_final_mix.inc"')
    source = replace_once(source,
        "Status QMoE::ComputeInternal(ComputeContext& context) const {",
        f'''Status QMoE::ComputeInternal(ComputeContext& context) const {{
  const int fc2_mix = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc2_mix == 0 || fc2_mix == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  key.push_back(fc1_activation);",
        "  key.push_back(fc1_activation);\n  key.push_back(fc2_mix);")
    source = replace_once(source,
        "  const size_t expected_commands = CanFuseFc1Activation(context, fc1_activation == 1) ? 4 : 5;",
        "  const size_t expected_commands = (CanFuseFc1Activation(context, fc1_activation == 1) ? 4 : 5) -\n"
        "      (CanFuseFc2FinalMix(context, fc2_mix == 1) ? 1 : 0);")
    start = source.index("    status = is_nvfp4_ || use_int4_cooperative",
                         source.index("    Tensor& fc2_outputs = temporary(4"))
    end = source.index("\n    return Status::OK();", start)
    return source[:start] + FUSED + source[start:end] + "\n    }\n" + source[end:]


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
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
    original = PARENT / "overlay/contrib_ops/webgpu/moe/qmoe.cc"
    source = overlay / "qmoe.cc"
    source.write_text(transform(original.read_text()))
    header = overlay / "qmoe.h"
    header.write_text(replace_once((original.parent / "qmoe.h").read_text(),
        "  struct ReplayEntry {",
        "  bool CanFuseFc2FinalMix(ComputeContext& context, bool enabled) const;\n"
        "  struct ReplayEntry {"))
    shutil.copy2(original.parent / "qmoe_fc1_activation.inc", overlay / "qmoe_fc1_activation.inc")
    shutil.copy2(ROOT / "qmoe_fc2_final_mix.inc", overlay / "qmoe_fc2_final_mix.inc")
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
    comparator.write_text((PARENT / "native_compare.cpp").read_text().replace(
        '"ORT_WEBGPU_QMOE_FC1_ACTIVATION"', f'"{FLAG}"'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, baseline_provider_sha256=parent["provider_sha256"],
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in (source, header, comparator,
            overlay / "qmoe_fc1_activation.inc", overlay / "qmoe_fc2_final_mix.inc")},
        comparator_sha256=sha(OUTPUT / "native_compare"),
        original_source_sha256=sha(original), original_header_sha256=sha(original.parent / "qmoe.h"),
        no_object_layout_changes=True, quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
