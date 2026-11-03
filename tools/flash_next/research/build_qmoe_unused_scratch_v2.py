import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from build_qmoe_unused_scratch import FLAG, PARENT, transform


OUTPUT = Path("/workspace/qwen38-int4-qmoe-unused-scratch/build-v2")


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
        "prefill_and_replay_allocation_order_preserved": True,
        "maximum_omitted_tensors_per_single_token_qmoe": 2,
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
