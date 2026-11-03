import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")


def build(output, source, transform, comparator, flag, metadata):
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    proof_path = PROOF / "baseline.qualification.json"
    proof = json.loads(proof_path.read_text())
    commands = json.loads((PROOF / "recovered-commands.json").read_text())
    component_path = source.parent / "build-manifest.json"
    component = json.loads(component_path.read_text())
    assert proof["all_152_original_objects_byte_identical"]
    assert len(commands) == len(proof["compile_commands"]) == 152
    assert parent["provider_sha256"] == "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"

    def original_object(path):
        value = Path(path)
        return value if value.is_absolute() else Path(parent["working_directory"]) / value

    for index, original in enumerate(commands):
        assert sha(original_object(original)) == sha(PROOF / "baseline" / f"object-{index}.o"), original
    for name, digest in proof["sdk_sha256"].items():
        assert sha(PROOF / name) == digest, name
    for path in (source, comparator):
        assert sha(path) == component["source_sha256"][path.name], path
    original, = [path for path, command in commands.items()
                 if command[command.index("-c") + 1] == str(source)]
    command = proof["compile_commands"][list(commands).index(original)].copy()
    output.mkdir(parents=True, exist_ok=False)
    obj = output / (source.name + ".o")
    command[command.index("-o") + 1] = str(obj)
    for option in ("-MF", "-MT"):
        if option in command:
            command[command.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert sha(obj) == sha(original_object(original)), "Original translation-unit rebuild mismatch"
    baseline_hash = sha(obj)
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Original provider relink mismatch"
    candidate = output / source.name
    candidate.write_text(transform(source.read_text()))
    command[command.index("-c") + 1] = str(candidate)
    command.insert(1, f"-fmacro-prefix-map={candidate}={source}")
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert link.count(original) == 1
    link[link.index(original)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert flag.encode() in library.read_bytes()
    comparator_copy = output / "native_compare.cpp"
    comparator_copy.write_text(comparator.read_text())
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator_copy),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(output / "native_compare")], check=True)
    inputs = [candidate, source, comparator, component_path, comparator_copy, proof_path,
              PROOF / "recovered-commands.json", *[PROOF / name for name in proof["sdk_sha256"]]]
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        all_152_original_source_objects_byte_identical=True, private_kernel_layout_only=True,
        no_object_layout_changes=True, original_translation_unit_sha256=baseline_hash,
        default_enabled=False, original_compute_and_fallbacks_unchanged=True,
        process_static_selector=True, comparator_sha256=sha(output / "native_compare"),
        compile_command=command, link_command=link,
        source_sha256={os.path.relpath(path, output): sha(path) for path in inputs})
    assert not report.keys() & metadata.keys(), "Metadata must not override provenance gates"
    report.update(metadata)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report
