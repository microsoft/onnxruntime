import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from ready_pipeline_state_transform import transform


ROOT = Path(__file__).parent
OUTPUT = Path("/workspace/qwen38-int4-ready-pipeline-state/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    proof = json.loads((PROOF / "baseline.qualification.json").read_text())
    if (parent["provider_sha256"] != "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"
            or sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]
            or not proof["all_152_original_objects_byte_identical"]):
        raise RuntimeError("Not the supported original carrier and frozen SDK proof")
    for name, digest in proof["sdk_sha256"].items():
        if sha(PROOF / name) != digest:
            raise RuntimeError(f"Changed frozen SDK input: {name}")
    original = COMPONENT / "overlay/core/providers/webgpu/webgpu_context.cc"
    for name, digest in component["overlay_source_sha256"].items():
        if sha(COMPONENT / "overlay" / name) != digest:
            raise RuntimeError(f"Changed original context overlay: {name}")
    command = next(c.copy() for c in component["compile_commands"]
                   if c[c.index("-c") + 1] == str(original))
    original_object = Path(command[command.index("-o") + 1])
    link = parent["link_command"].copy()
    if link.count(str(original_object)) != 1:
        raise RuntimeError("Original context object is not unique in supported carrier")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))

    def compile_object(source, obj):
        invocation = command.copy()
        invocation[invocation.index("-c") + 1] = str(source)
        invocation[invocation.index("-o") + 1] = str(obj)
        for option in ("-MF", "-MT"):
            if option in invocation:
                invocation[invocation.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
        subprocess.run(invocation, cwd=parent["working_directory"], env=environment, check=True)
        return invocation

    baseline = OUTPUT / "original-webgpu_context.cc.o"
    compile_object(original, baseline)
    if sha(baseline) != sha(original_object):
        raise RuntimeError("Original context object does not reproduce")
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Supported original provider does not reproduce")
    source = OUTPUT / "webgpu_context.cc"
    source.write_text(f'#line 1 "{original}"\n' + transform(original.read_text()))
    obj = OUTPUT / "webgpu_context.cc.o"
    selected_command = compile_object(source, obj)
    link[link.index(str(original_object))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    inputs = (Path(__file__), ROOT / "ready_pipeline_state_transform.py",
              PARENT / "build-manifest.json", COMPONENT / "build-manifest.json",
              PROOF / "baseline.qualification.json", original, original_object,
              baseline, source, *[PROOF / name for name in proof["sdk_sha256"]],
              *[COMPONENT / "overlay" / name for name in component["overlay_source_sha256"]])
    report = {
        "status": "built_not_qualified", "provider_sha256": sha(library),
        "baseline_provider_sha256": parent["provider_sha256"], "baseline_byte_identical": True,
        "original_context_object_byte_identical": True,
        "control_definition": "original3a_scoped_redundant_pipeline_state_skip",
        "shader_math_and_all_other_objects_unchanged": True,
        "legacy_event_fields_unchanged": True, "no_object_layout_changes": True,
        "no_global_profiler_or_hardware_setting_changes": True,
        "compile_command": selected_command, "link_command": link,
        "working_directory": parent["working_directory"],
        "source_sha256": {str(path): sha(path) for path in inputs},
        "performance_qualified": False, "quality_qualified": False,
    }
    with (OUTPUT / "build-manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "original_context_object_byte_identical")}), flush=True)


if __name__ == "__main__":
    main()
