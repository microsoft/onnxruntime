import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_qmoe_unused_scratch import FC1_CONDITION, FC2_CONDITION, PARENT


OUTPUT = Path("/workspace/qwen38-int4-qmoe-early-scratch/build-v1")
FLAG = "ORT_WEBGPU_QMOE_EARLY_SCRATCH"
SELECTOR = """
    static const int early_scratch = [] {
      const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_EARLY_SCRATCH", 0);
      ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_QMOE_EARLY_SCRATCH must be 0 or 1");
      return value;
    }();
"""
PREALLOCATE = (
    "\n    if (early_scratch == 1 && scratch == nullptr && num_tokens == 1 &&\n"
    "        " + FC1_CONDITION + " &&\n"
    "        " + FC2_CONDITION + ") {\n"
    "      temporary(2, dtype, TensorShape({static_cast<int64_t>(routed_rows), fc1_output_size}));\n"
    "      temporary(3, dtype, TensorShape({static_cast<int64_t>(routed_rows), moe_params.inter_size}));\n"
    "      temporary(4, dtype, TensorShape({static_cast<int64_t>(routed_rows), N_fc2}));\n"
    "    }\n"
)


def transform(source):
    source = replace_once(source, "    std::array<std::optional<Tensor>, 5> local_scratch;",
                          SELECTOR + "    std::array<std::optional<Tensor>, 5> local_scratch;")
    anchor = "    Tensor& indirect_experts = temporary(1, dtype_uint32, indirect_experts_shape);\n"
    return replace_once(source, anchor, anchor + PREALLOCATE)


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    original = PARENT / "qmoe.cc"
    original_object = PARENT / "qmoe.cc.o"
    command = list(parent["compile_command"])
    if command[command.index("-c") + 1] != str(original):
        raise RuntimeError("Carrier compile entry does not name its actual QMoE source")
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
        "all_five_allocations_and_zero_initialization_preserved": True,
        "prefill_replay_and_unfused_allocation_order_preserved": True,
        "source_sha256": {str(path): sha(path) for path in
                          (Path(__file__), Path(__file__).with_name("build_qmoe_unused_scratch.py"),
                           source, original)},
        "performance_qualified": False,
        "quality_qualified": False,
    }
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "original_qmoe_object_byte_identical")}))


if __name__ == "__main__":
    main()
