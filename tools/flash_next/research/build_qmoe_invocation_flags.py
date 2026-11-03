import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-invocation-flags/build-v1")
HEADER = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe/qmoe.h")
FLAG = "ORT_WEBGPU_QMOE_INVOCATION_FLAGS"
FIELDS = (
    ("expert_bitcast", "ORT_WEBGPU_EXPERT_BITCAST_UNPACK"),
    ("extended_prefill", "ORT_WEBGPU_QMOE_EXTENDED_PREFILL"),
    ("prefill256", "ORT_WEBGPU_QMOE_PREFILL256"),
    ("prefill512", "ORT_WEBGPU_QMOE_PREFILL512"),
    ("prefill1024", "ORT_WEBGPU_QMOE_PREFILL1024"),
    ("grouped_prefill", "ORT_WEBGPU_QMOE_GROUPED_PREFILL"),
    ("hierarchical_gate", "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE"),
    ("fc2_mix", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX"),
    ("fc1_activation", "ORT_WEBGPU_QMOE_FC1_ACTIVATION"),
)
STRUCT = (
    "  struct InvocationFlags {\n"
    + "".join(f"    int {field};\n" for field, _ in FIELDS)
    + "  };\n"
)
SELECTOR = """
  static const int reuse_invocation_flags = [] {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INVOCATION_FLAGS", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_QMOE_INVOCATION_FLAGS must be 0 or 1");
    return value;
  }();
  const InvocationFlags invocation_flags{expert_bitcast, extended_prefill, prefill256,
      prefill512, prefill1024, grouped_prefill, hierarchical_gate, fc2_mix, fc1_activation};
  const InvocationFlags* selected_flags = reuse_invocation_flags == 1 ? &invocation_flags : nullptr;
"""


def transform(source, header):
    header = replace_once(
        header, "  Status ComputeWithWorkspace(ComputeContext& context,\n",
        STRUCT + "  Status ComputeWithWorkspace(ComputeContext& context,\n")
    header = replace_once(
        header, "std::array<std::optional<Tensor>, 5>* scratch) const;",
        "std::array<std::optional<Tensor>, 5>* scratch,\n"
        "                              const InvocationFlags* invocation_flags) const;")
    source = replace_once(
        source, '  const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0);',
        SELECTOR + '  const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0);')
    if source.count("ComputeWithWorkspace(context, nullptr)") != 2:
        raise RuntimeError("Unexpected ordinary/fallback workspace call count")
    source = source.replace("ComputeWithWorkspace(context, nullptr)",
                            "ComputeWithWorkspace(context, nullptr, selected_flags)")
    source = replace_once(
        source, "ComputeWithWorkspace(context, &replay_scratch_)",
        "ComputeWithWorkspace(context, &replay_scratch_, selected_flags)")
    source = replace_once(
        source, "std::array<std::optional<Tensor>, 5>* scratch) const {",
        "std::array<std::optional<Tensor>, 5>* scratch,\n"
        "                                 const InvocationFlags* invocation_flags) const {")
    boundary = source.index("Status QMoE::ComputeWithWorkspace(")
    head, body = source[:boundary], source[boundary:]
    for field, flag in FIELDS:
        original = f'ParseEnvironmentVariableWithDefault<int>("{flag}", 0)'
        body = replace_once(
            body, original,
            f"(invocation_flags != nullptr ? invocation_flags->{field} : {original})")
    return head + body, header


def main():
    manifest_path = PARENT / "build-manifest.json"
    parent = json.loads(manifest_path.read_text())
    original = PARENT / "qmoe.cc"
    original_object = PARENT / "qmoe.cc.o"
    command = list(parent["compile_command"])
    if command[command.index("-c") + 1] != str(original):
        raise RuntimeError("Wrong original QMoE compile entry")
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Changed supported carrier")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    baseline_object = OUTPUT / "qmoe.baseline.cc.o"
    command[command.index("-o") + 1] = str(baseline_object)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    if sha(baseline_object) != sha(original_object):
        raise RuntimeError("Exact original QMoE object proof failed")
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = list(parent["link_command"])
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Original provider relink proof failed")
    library.unlink()
    source = OUTPUT / "qmoe.cc"
    overlay = OUTPUT / "overlay/contrib_ops/webgpu/moe/qmoe.h"
    transformed_source, transformed_header = transform(original.read_text(), HEADER.read_text())
    overlay.parent.mkdir(parents=True)
    source.write_text(transformed_source)
    overlay.write_text(transformed_header)
    obj = OUTPUT / "qmoe.cc.o"
    command.insert(1, "-I" + str(OUTPUT / "overlay"))
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
        "compile_commands": [command], "link_command": link,
        "working_directory": parent["working_directory"],
        "flag": FLAG, "default_enabled": False,
        "selector_scope": "invocation_local_values_refresh_every_compute",
        "no_persistent_value_cache": True,
        "no_shader_changes": True, "no_object_layout_changes": True,
        "entry_parse_order_range_errors_and_replay_keys_unchanged": True,
        "source_sha256": {str(path): sha(path) for path in
                          (Path(__file__), source, overlay, original, HEADER, manifest_path)},
        "performance_qualified": False, "quality_qualified": False,
    }
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("provider_sha256", "baseline_byte_identical",
                       "original_qmoe_object_byte_identical")}))


if __name__ == "__main__":
    main()
