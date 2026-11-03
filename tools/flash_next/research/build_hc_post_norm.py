import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", choices=("v4", "v5", "v6", "v7", "v8"), default="v8")
    args = parser.parse_args()
    root = Path(__file__).parent
    prior = json.loads(Path("/workspace/qwen38-int4-quickgelu-next/build/build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-hc-post-norm/build-" + args.revision)
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    link = prior["link_command"].copy()
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    assert sha(library) == prior["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = output / "hc_post_norm_overlay.cc"
    shutil.copyfile(root / source.name, source)
    registry = output / "webgpu_contrib_kernels.cc"
    text = Path("/workspace/qwen38-int4-quickgelu-next/build/webgpu_contrib_kernels.cc").read_text()
    anchor = "    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, QuickGeluMulF16)>,"
    assert text.count(anchor) == 1
    registry.write_text(text.replace(anchor, anchor + "\n" + "\n".join(
        anchor.replace("QuickGeluMulF16", op) for op in ("HCPostNormF16", "HCSharedPostNormF16"))))
    commands = []
    for path in (source, registry):
        obj = output / (path.name + ".o")
        command = prior["compile_commands"][0].copy()
        command[command.index("-c") + 1] = str(path)
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=prior["working_directory"], env=environment, check=True)
        commands.append(command)
        if path == registry:
            matches = [p for p in link if p.endswith("webgpu_contrib_kernels.cc.o")]
            assert len(matches) == 1
            link[link.index(matches[0])] = str(obj)
        else:
            link.append(str(obj))
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    native_command = ["c++", "-O3", "-std=c++20", str(root / "hc_fusion_compare.cpp"),
                      "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                      "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                      "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                      "-lonnxruntime", "-o", str(output / "native_compare")]
    subprocess.run(native_command, env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
                  compile_commands=commands, link_command=link, working_directory=prior["working_directory"],
                  source_sha256={str(p): sha(p) for p in (source, registry)},
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
