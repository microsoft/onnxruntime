import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-unused-scratch/build-v1")
FLAG = "ORT_WEBGPU_QMOE_LAZY_UNUSED_SCRATCH"
SELECTOR = """
    static const int lazy_unused_scratch = [] {
      const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_LAZY_UNUSED_SCRATCH", 0);
      ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_QMOE_LAZY_UNUSED_SCRATCH must be 0 or 1");
      return value;
    }();
    const bool omit_unused_scratch =
        lazy_unused_scratch == 1 && scratch == nullptr && moe_params.num_rows == 1;
"""
FC1_CONDITION = """CanFuseFc1Activation(context,
          ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC1_ACTIVATION", 0) == 1)"""
FC2_CONDITION = """CanFuseFc2FinalMix(context,
          ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", 0) == 1)"""


def transform(source):
    source = replace_once(source, "    std::array<std::optional<Tensor>, 5> local_scratch;",
                          SELECTOR + "    std::array<std::optional<Tensor>, 5> local_scratch;")
    source = replace_once(source, "    Tensor& fc1_outputs = temporary(2, dtype, fc1_output_shape);",
        "    const bool fuse_fc1_activation = " + FC1_CONDITION + ";\n"
        "    if (!omit_unused_scratch || !fuse_fc1_activation) {\n"
        "      temporary(2, dtype, fc1_output_shape);\n"
        "    }")
    source = replace_once(source, "    if (" + FC1_CONDITION + ") {",
                          "    if (fuse_fc1_activation) {")
    source = replace_fc1_fallback(source)
    source = replace_once(source, "    Tensor& fc2_outputs = temporary(4, dtype, fc2_output_shape);",
        "    const bool fuse_fc2_final_mix = " + FC2_CONDITION + ";\n"
        "    if (!omit_unused_scratch || !fuse_fc2_final_mix) {\n"
        "      temporary(4, dtype, fc2_output_shape);\n"
        "    }")
    source = replace_once(source, "    if (" + FC2_CONDITION + ") {",
                          "    if (fuse_fc2_final_mix) {")
    anchor = """    } else {
    status = is_nvfp4_ || use_int4_cooperative
                 ? ApplyBlockQuantizedExpertMatMul(context, &fc1_activated,"""
    source = replace_once(source, anchor, """    } else {
    Tensor& fc2_outputs = temporary(4, dtype, fc2_output_shape);
    status = is_nvfp4_ || use_int4_cooperative
                 ? ApplyBlockQuantizedExpertMatMul(context, &fc1_activated,""")
    return source


def replace_fc1_fallback(source):
    anchor = """    } else {
    status = is_nvfp4_ || use_int4_cooperative
                 ? ApplyBlockQuantizedExpertMatMul(context, hidden_state,"""
    return replace_once(source, anchor, """    } else {
    Tensor& fc1_outputs = temporary(2, dtype, fc1_output_shape);
    status = is_nvfp4_ || use_int4_cooperative
                 ? ApplyBlockQuantizedExpertMatMul(context, hidden_state,""")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    original = PARENT / "qmoe.cc"
    original_object = PARENT / "qmoe.cc.o"
    command = next(command.copy() for command in parent["compile_commands"]
                   if command[command.index("-c") + 1] == str(original))
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Supported carrier changed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    baseline_object = OUTPUT / "qmoe.baseline.cc.o"
    command[command.index("-o") + 1] = str(baseline_object)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    if sha(baseline_object) != sha(original_object):
        raise RuntimeError("Exact original QMoE object proof failed; no GPU admission")
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = list(parent["link_command"])
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Original carrier relink proof failed")
    library.unlink()
    source = OUTPUT / "qmoe.cc"
    source.write_text(transform(original.read_text()))
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    if link.count(str(original_object)) != 1:
        raise RuntimeError("Unexpected original QMoE object count")
    link[link.index(str(original_object))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    report = {
        "status": "built_not_qualified",
        "provider_sha256": sha(library),
        "baseline_provider_sha256": parent["provider_sha256"],
        "baseline_byte_identical": True,
        "original_qmoe_object_byte_identical": True,
        "compile_commands": [command],
        "link_command": link,
        "working_directory": parent["working_directory"],
        "flag": FLAG,
        "default_enabled": False,
        "selector_scope": "process_static_fresh_process_required",
        "no_shader_changes": True,
        "no_object_layout_changes": True,
        "prefill_and_replay_allocation_order_preserved": True,
        "maximum_omitted_tensors_per_single_token_qmoe": 2,
        "source_sha256": {str(path): sha(path) for path in (Path(__file__), source, original)},
        "performance_qualified": False,
        "quality_qualified": False,
    }
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "original_qmoe_object_byte_identical")}))


if __name__ == "__main__":
    main()
