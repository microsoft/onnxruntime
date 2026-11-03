import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    prior = json.loads((repo / "build-int4-bindgroup-cache/native-v4/build-manifest.json").read_text())
    output = repo / "build-int4-dynamic-uniforms"
    output.mkdir(exist_ok=True)
    consumers = {
        "core/providers/webgpu/allocator.cc",
        "core/providers/webgpu/buffer_manager.cc",
        "core/providers/webgpu/compute_context.cc",
        "core/providers/webgpu/data_transfer.cc",
        "core/providers/webgpu/ep/allocator.cc",
        "core/providers/webgpu/ep/factory.cc",
        "core/providers/webgpu/ep/sync_stream.cc",
        "core/providers/webgpu/program_manager.cc",
        "core/providers/webgpu/webgpu_context.cc",
        "core/providers/webgpu/webgpu_execution_provider.cc",
        "core/providers/webgpu/webgpu_provider_factory.cc",
    }
    baseline = output / "baseline"
    baseline.mkdir(exist_ok=True)
    link = prior["link_command"].copy()
    baseline_library = baseline / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(baseline_library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    if digest(baseline_library) != prior["provider_sha256"]:
        raise RuntimeError("Unchanged-object link does not reproduce the qualified v4 provider")
    replacements = {}
    commands = []
    for original in prior["link_command"]:
        if "/build-int4-uniform-arena/" in original and original.endswith(".cc.o"):
            relative = Path(original).name.removesuffix(".o").replace("__", "/")
            source = repo / "onnxruntime" / relative
        elif original.endswith("webgpu_context.cc.o"):
            source = repo / "onnxruntime/core/providers/webgpu/webgpu_context.cc"
        else:
            continue
        if str(source.relative_to(repo / "onnxruntime")) not in consumers:
            continue
        if not source.is_file():
            raise RuntimeError(f"Missing source for affected command-state object: {source}")
        obj = output / (str(source.relative_to(repo)).replace("/", "__") + ".o")
        command = prior["compile_command"].copy()
        command[command.index("-c") + 1] = str(source)
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=prior["working_directory"], check=True)
        replacements[original] = str(obj)
        commands.append(command)
    if len(replacements) != 11:
        raise RuntimeError(f"Expected all 11 command-state consumers, found {len(replacements)}")
    link = [replacements.get(arg, arg) for arg in prior["link_command"]]
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    sources = [Path(command[command.index("-c") + 1]) for command in commands]
    sources.extend(repo / "onnxruntime/core/providers/webgpu" / name
                   for name in ("webgpu_context.h", "program_manager.h"))
    report = {
        "status": "built_not_GPU_qualified",
        "baseline_byte_identical": True,
        "baseline_provider_sha256": prior["provider_sha256"],
        "provider_sha256": digest(library),
        "source_sha256": {str(path): digest(path) for path in sources},
        "compile_commands": commands,
        "link_command": link,
        "working_directory": prior["working_directory"],
        "abi_note": "All 11 objects from the prior command-state/uniform-arena rebuild are recompiled.",
    }
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "baseline_byte_identical", "provider_sha256", "abi_note")}))


if __name__ == "__main__":
    main()
