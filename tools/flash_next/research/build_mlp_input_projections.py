import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-qmoe-hierarchical-gate/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-hc-projection-silu/build-v1")
REGISTRY = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-mlp-input-projections/build-v2")
FLAG = "ORT_WEBGPU_MLP_INPUT_PROJECTIONS"


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    registry = json.loads((REGISTRY / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = OUTPUT / "mlp_input_projections.cc"
    source.write_text((ROOT / source.name).read_text())
    registry_source = OUTPUT / "webgpu_contrib_kernels.cc"
    anchor = "    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, HCProjectionSiLUF16)>,"
    registry_source.write_text(replace_once((REGISTRY / registry_source.name).read_text(), anchor,
        anchor + "\n    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MlpInputProjectionsF16)>,"))
    commands = []
    for template, path in ((component["compile_commands"][0], source),
                           (registry["compile_commands"][1], registry_source)):
        command = template.copy()
        command[1:1] = ["-I/workspace/qwen38-int4-static-node-replay/build-v8/overlay"]
        command[command.index("-c") + 1] = str(path)
        obj = OUTPUT / (path.name + ".o")
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
        commands.append(command)
        if path == registry_source:
            original_obj = str(REGISTRY / obj.name)
            assert link.count(original_obj) == 1
            link[link.index(original_obj)] = str(obj)
        else:
            link.append(str(obj))
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(replace_once((COMPONENT / comparator.name).read_text(),
        'setenv("ORT_WEBGPU_HC_PROJECTION_SILU", value, 1)', f'setenv("{FLAG}", value, 1)'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=commands, link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, registry_source, comparator)},
        comparator_sha256=sha(OUTPUT / "native_compare"), no_runtime_object_layout_changes=True,
        compound_fallback_not_replay_cached=True, quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
