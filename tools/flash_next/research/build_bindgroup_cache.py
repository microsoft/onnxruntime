import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    prior = json.loads((repo / "build-int4-matmul-gemv-n96/build-manifest.json").read_text())
    output = repo / "build-int4-bindgroup-cache"
    output.mkdir(exist_ok=True)
    baseline = output / "baseline"
    baseline.mkdir(exist_ok=True)
    link = prior["link_command"].copy()
    baseline_library = baseline / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(baseline_library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    if digest(baseline_library) != prior["provider_sha256"]:
        raise RuntimeError("Baseline does not reproduce the qualified provider")
    source = repo / "onnxruntime/core/providers/webgpu/webgpu_context.cc"
    header = source.with_suffix(".h")
    command = prior["compile_command"].copy()
    command[command.index("-c") + 1] = str(source)
    obj = output / "webgpu_context.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=prior["working_directory"], check=True)
    link = prior["link_command"].copy()
    originals = [arg for arg in link if arg.endswith("webgpu_context.cc.o")]
    if len(originals) != 1:
        raise RuntimeError(f"Expected one context object, found {originals}")
    link[link.index(originals[0])] = str(obj)
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    report = {
        "status": "built_not_GPU_qualified",
        "baseline_byte_identical": True,
        "baseline_provider_sha256": prior["provider_sha256"],
        "provider_sha256": digest(library),
        "source_sha256": {str(path): digest(path) for path in (source, header)},
        "compile_command": command,
        "link_command": link,
        "working_directory": prior["working_directory"],
    }
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "baseline_byte_identical", "provider_sha256")}))


if __name__ == "__main__":
    main()
