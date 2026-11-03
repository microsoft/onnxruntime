import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    prior = json.loads((repo / "build-int4-dynamic-uniforms/build-manifest.json").read_text())
    output = repo / "build-int4-hyper-skinny"
    output.mkdir(exist_ok=True)
    library = output / "libonnxruntime_providers_webgpu.so"
    if library.exists():
        raise RuntimeError("Refusing to overwrite a previously built provider")
    baseline_dir = output / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline = baseline_dir / library.name
    link = prior["link_command"].copy()
    link[link.index("-o") + 1] = str(baseline)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    if digest(baseline) != prior["provider_sha256"]:
        raise RuntimeError("Unchanged objects do not reproduce the qualified dynamic-uniform provider")
    baseline.unlink()
    baseline_dir.rmdir()
    old = [arg for arg in prior["link_command"] if arg.endswith("__matmul_nbits.cc.o")]
    if len(old) != 1:
        raise RuntimeError(f"Expected one dense MatMulNBits object, found {old}")
    source = repo / "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc"
    obj = output / "matmul_nbits.cc.o"
    compile_command = prior["compile_commands"][0].copy()
    compile_command[compile_command.index("-c") + 1] = str(source)
    compile_command[compile_command.index("-o") + 1] = str(obj)
    subprocess.run(compile_command, cwd=prior["working_directory"], check=True)
    link = [str(obj) if arg == old[0] else arg for arg in prior["link_command"]]
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    shutil.copyfile(source, output / source.name)
    report = {
        "status": "built_not_GPU_qualified",
        "baseline_byte_identical": True,
        "baseline_provider_sha256": prior["provider_sha256"],
        "provider_sha256": digest(library),
        "source_sha256": {str(source): digest(source)},
        "compile_command": compile_command,
        "link_command": link,
        "working_directory": prior["working_directory"],
        "abi_note": "Only MatMulNBits changes; all eleven qualified command-state objects retained.",
    }
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "provider_sha256")}), flush=True)


if __name__ == "__main__":
    main()
