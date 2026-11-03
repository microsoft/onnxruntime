import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
BUILD = Path("/workspace/onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release")
OUTPUT = Path("/workspace/qwen38-int4-dawn-compute-scope-reuse/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2")
FLAG = "ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE"

WRAPPER = """
void ComputePassEncoder::AddDispatchSyncScope() {
#if ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE
    thread_local bool inUse = false;
    if (inUse) {
        AddDispatchSyncScope(SyncScopeUsageTracker{});
        return;
    }
    struct ResetUse {
        bool& flag;
        ~ResetUse() { flag = false; }
    };
    inUse = true;
    ResetUse reset{inUse};
    thread_local SyncScopeUsageTracker scope;
    AddDispatchSyncScopeInternal(scope);
#else
    AddDispatchSyncScope(SyncScopeUsageTracker{});
#endif
}

void ComputePassEncoder::AddDispatchSyncScope(SyncScopeUsageTracker scope) {
    AddDispatchSyncScopeInternal(scope);
}

"""


def transform(source, header):
    source = replace_once(source, "namespace dawn::native {\n",
        "#ifndef ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE\n"
        "#define ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE 0\n"
        "#endif\n"
        "#if ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE != 0 && ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE != 1\n"
        '#error "ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE must be 0 or 1"\n'
        "#endif\n\nnamespace dawn::native {\n")
    source = replace_once(source,
        "void ComputePassEncoder::AddDispatchSyncScope(SyncScopeUsageTracker scope) {",
        WRAPPER + "void ComputePassEncoder::AddDispatchSyncScopeInternal(SyncScopeUsageTracker& scope) {")
    header = replace_once(header,
        "    void AddDispatchSyncScope(SyncScopeUsageTracker scope = {});",
        "    void AddDispatchSyncScope();\n"
        "    void AddDispatchSyncScope(SyncScopeUsageTracker scope);\n"
        "    void AddDispatchSyncScopeInternal(SyncScopeUsageTracker& scope);")
    return source, header


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Supported carrier changed")
    entries = json.loads((BUILD / "compile_commands.json").read_text())
    entry = next(item for item in entries if item["file"].endswith("/ComputePassEncoder.cpp"))
    original = Path(entry["file"])
    original_header = original.with_suffix(".h")
    obj = BUILD / "_deps/dawn-build/src/dawn/native/CMakeFiles/dawn_native_objects.dir/ComputePassEncoder.cpp.o"
    archive = BUILD / "_deps/dawn-build/src/dawn/native/libdawn_native.a"
    original_archive_sha = sha(archive)
    original_obj_sha = sha(obj)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    command = shlex.split(entry["command"])
    baseline_object = OUTPUT / "ComputePassEncoder.baseline.cpp.o"
    command[command.index("-o") + 1] = str(baseline_object)
    subprocess.run(command, cwd=entry["directory"], env=environment, check=True)
    if sha(baseline_object) != original_obj_sha:
        raise RuntimeError("Exact original Dawn object proof failed; no GPU admission")
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = list(parent["link_command"])
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Original carrier relink proof failed")
    library.unlink()
    source_text, header_text = transform(original.read_text(), original_header.read_text())
    source = OUTPUT / original.name
    header = OUTPUT / "overlay/src/dawn/native/ComputePassEncoder.h"
    header.parent.mkdir(parents=True)
    source.write_text(source_text)
    header.write_text(header_text)
    candidate_object = OUTPUT / "ComputePassEncoder.cpp.o"
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(candidate_object)
    command[1:1] = [f"-I{OUTPUT / 'overlay'}", f"-D{FLAG}=1"]
    subprocess.run(command, cwd=entry["directory"], env=environment, check=True)
    private_archive = OUTPUT / archive.name
    shutil.copy2(archive, private_archive)
    members = subprocess.check_output(["ar", "t", str(private_archive)], text=True).splitlines()
    if members.count(candidate_object.name) != 1:
        raise RuntimeError("Original archive does not contain exactly one Dawn compute-pass object")
    subprocess.run(["ar", "rcsD", str(private_archive), str(candidate_object)], check=True)
    replacements = 0
    for i, value in enumerate(link):
        if value == "_deps/dawn-build/src/dawn/native/libdawn_native.a":
            link[i] = str(private_archive)
            replacements += 1
    if replacements != 2:
        raise RuntimeError("Original carrier Dawn archive references changed")
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    if sha(archive) != original_archive_sha or sha(obj) != original_obj_sha:
        raise RuntimeError("Original Dawn build assets changed")
    comparator = OUTPUT / "native_compare"
    shutil.copy2(COMPONENT / "native_compare", comparator)
    sources = [Path(__file__), source, header, original, original_header,
               original.with_name("PassResourceUsageTracker.cpp"),
               original.with_name("PassResourceUsageTracker.h")]
    report = {
        "status": "built_not_qualified",
        "provider_sha256": sha(library),
        "baseline_provider_sha256": parent["provider_sha256"],
        "baseline_byte_identical": True,
        "original_dawn_object_byte_identical": True,
        "original_dawn_object_sha256": original_obj_sha,
        "original_dawn_archive_sha256": original_archive_sha,
        "private_dawn_archive_sha256": sha(private_archive),
        "compile_commands": [command],
        "link_command": link,
        "working_directory": parent["working_directory"],
        "compile_time_flag": FLAG,
        "default_enabled": False,
        "candidate_compile_time_enabled": True,
        "supported_carrier_unchanged": True,
        "no_object_layout_changes": True,
        "indirect_scope_keeps_owned_by_value_tracker": True,
        "reentrant_scope_keeps_owned_by_value_tracker": True,
        "validation_and_synchronization_body_unchanged": True,
        "comparator_sha256": sha(comparator),
        "source_sha256": {str(path): sha(path) for path in sources},
        "headline_eligible": False,
        "quality_qualified": False,
    }
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "original_dawn_object_byte_identical")}))


if __name__ == "__main__":
    main()
