import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", choices=("v2",), default="v2")
    revision = parser.parse_args().revision
    root = Path(__file__).parent
    repos = Path("/workspace")
    prior_dir = repos / "qwen38-int4-hc-projection-silu/build-v1"
    prior = json.loads((prior_dir / "build-manifest.json").read_text())
    output = repos / ("qwen38-int4-hc-up-premix/build-" + revision)
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = output / "libonnxruntime_providers_webgpu.so"
    link = prior["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    assert sha(library) == prior["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = output / "hc_up_premix.cc"
    shutil.copy2(root / source.name, source)
    registry = output / "webgpu_contrib_kernels.cc"
    anchor = "    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, HCProjectionSiLUF16)>,"
    registry.write_text(replace_once((prior_dir / registry.name).read_text(), anchor,
        anchor + "\n" + anchor.replace("HCProjectionSiLUF16", "HCUpPreMixF16")))
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
    comparator = output / "native_compare.cpp"
    text = (root / "static_node_replay_compare.cpp").read_text()
    comparator.write_text(replace_once(text, 'setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1)',
                                      'setenv("ORT_WEBGPU_HC_UP_PREMIX", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(output / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
        compile_commands=commands, link_command=link, working_directory=prior["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, registry, comparator)},
        quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
