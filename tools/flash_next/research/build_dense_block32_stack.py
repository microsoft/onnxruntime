import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import sha


PARENT = Path("/workspace/qwen38-int4-dsa-query-cache/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-dense-block32/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-dense-block32/build-v2")
ORIGINAL_OBJECT = "/workspace/qwen38-int4-dense-tile4/build-v1/matmul_nbits.cc.o"


def combined_link(parent, component, output):
    command = parent["link_command"].copy()
    assert command.count(ORIGINAL_OBJECT) == 1
    assert component["link_command"].count(str(COMPONENT / "matmul_nbits.cc.o")) == 1
    command[command.index(ORIGINAL_OBJECT)] = str(COMPONENT / "matmul_nbits.cc.o")
    command[command.index("-o") + 1] = str(output)
    return command


def main():
    manifests = [json.loads((directory / "build-manifest.json").read_text())
                 for directory in (PARENT, COMPONENT)]
    parent, component = manifests
    for directory, manifest in zip((PARENT, COMPONENT), manifests):
        assert sha(directory / "libonnxruntime_providers_webgpu.so") == manifest["provider_sha256"]
        for name, digest in manifest["source_sha256"].items():
            assert sha(directory / name) == digest
    assert parent["fp16_only"] and parent["fp32_original_fallback_preserved"]
    assert parent["baseline_provider_sha256"] == component["baseline_provider_sha256"]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    for directory, manifest in zip((PARENT, COMPONENT), manifests):
        command = manifest["link_command"].copy()
        command[command.index("-o") + 1] = str(library)
        subprocess.run(command, cwd=manifest["working_directory"], env=environment, check=True)
        assert sha(library) == manifest["provider_sha256"], f"Frozen relink mismatch: {directory}"
        library.unlink()
    command = combined_link(parent, component, library)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    shutil.copy2(COMPONENT / "native_compare", OUTPUT / "native_compare")
    sources = {}
    for directory, manifest in zip((PARENT, COMPONENT), manifests):
        for name, digest in manifest["source_sha256"].items():
            sources[os.path.relpath(directory / name, OUTPUT)] = digest
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        dense_component_relink_byte_identical=True, query_cache_object_unchanged=True,
        no_new_shader_or_arithmetic_changes=True, source_sha256=sources,
        link_command=command, compile_commands=component["compile_commands"],
        working_directory=parent["working_directory"],
        comparator_sha256=sha(OUTPUT / "native_compare"),
        dense_object_sha256=sha(COMPONENT / "matmul_nbits.cc.o"),
        component_manifests_sha256={str(directory / "build-manifest.json"):
            sha(directory / "build-manifest.json") for directory in (PARENT, COMPONENT)},
        quality_qualified=False, headline_eligible=False)
    with (OUTPUT / "build-manifest.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("status", "provider_sha256", "baseline_byte_identical")}), flush=True)


if __name__ == "__main__":
    main()
