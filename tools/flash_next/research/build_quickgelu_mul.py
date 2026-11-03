import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    evidence = Path(__file__).parent
    output = Path("/workspace/qwen38-int4-quickgelu-next/build")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "build-manifest.json").exists():
        raise FileExistsError("Never overwrite an existing build")
    prior = json.loads((repo / "build-int4-shared-post-v4/build-manifest.json").read_text())
    link = prior["link_command"].copy()
    baseline_dir = output / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline = baseline_dir / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(baseline)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    assert sha(baseline) == prior["provider_sha256"], "Baseline relink mismatch"
    baseline.unlink()
    baseline_dir.rmdir()
    for name in ("quickgelu_mul_overlay.cc", "quickgelu_mul_overlay.h"):
        shutil.copyfile(evidence / name, output / name)
    registry = repo / "onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc"
    text = registry.read_text()
    old_line = "    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, SplitScaledSiLUF16)>,"
    assert text.count(old_line) == 1
    copied_registry = output / registry.name
    copied_registry.write_text(text.replace(old_line, old_line + "\n" +
        old_line.replace("SplitScaledSiLUF16", "QuickGeluMulF16")))
    commands = []
    for source in (output / "quickgelu_mul_overlay.cc", copied_registry):
        obj = output / (source.name + ".o")
        command = prior["compile_commands"][0].copy()
        command[command.index("-c") + 1] = str(source)
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=prior["working_directory"], check=True)
        commands.append(command)
        if source == copied_registry:
            old = [p for p in link if p.endswith("webgpu_contrib_kernels.cc.o")]
            assert len(old) == 1, old
            link = [str(obj) if p == old[0] else p for p in link]
        else:
            link.append(str(obj))
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    report = dict(baseline_byte_identical=True, baseline_provider_sha256=prior["provider_sha256"],
                  provider_sha256=sha(library), compile_commands=commands, link_command=link,
                  working_directory=prior["working_directory"],
                  source_sha256={str(p): sha(p) for p in (
                      output / "quickgelu_mul_overlay.cc", output / "quickgelu_mul_overlay.h",
                      copied_registry)}, quality_qualified=False,
                  reference="ORT production QuickGelu(alpha=1) then Mul; not raw unfused sigmoid")
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
