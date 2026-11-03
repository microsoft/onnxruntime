import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-qmoe-fusions-map/build-v2")
PROJECTION = Path("/workspace/qwen38-int4-hc-projection-silu/build-v1")
REGISTRY = Path("/workspace/qwen38-int4-hc-up-premix/build-v2")
STATIC = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
OUTPUT = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2")
FLAG = "ORT_WEBGPU_HC_POST_NORM_PROJECTION"


def program_class(source, name, following, replacement):
    start = source.index(f"class {name} final :")
    stop = source.index(f"\nclass {following}", start)
    return source[start:stop].replace(name, replacement)


def source_text(norm, projection):
    includes = projection[:projection.index("namespace onnxruntime::contrib::webgpu")]
    includes += '#include "contrib_ops/cpu/hyper_connection_helper.h"\n'
    parts = [includes, "namespace onnxruntime::contrib::webgpu {\nusing namespace onnxruntime::webgpu;\n"]
    parts.append(program_class(norm, "HCPostNormProgram", "HCPostNormF16",
                               "HCPostNormProjectionBaselineNormProgram"))
    parts.append(program_class(projection, "HCProjectionSiLUProgram", "HCProjectionSiLUF16",
                               "HCPostNormProjectionBaselineMatMulProgram"))
    parts += [(ROOT / name).read_text() for name in
              ("hc_post_norm_projection_program.inc", "hc_post_norm_projection_kernel.inc")]
    return "\n".join(parts) + "\n}\n"


def registry_text(source):
    anchor = "    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, HCProjectionSiLUF16)>,"
    extra = "\n".join(
        f"    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, {name})>,"
        for name in ("HCPostNormProjectionF16", "HCSharedPostNormProjectionF16"))
    return replace_once(source, anchor, anchor + "\n" + extra)


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((PROJECTION / "build-manifest.json").read_text())
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
    source = OUTPUT / "hc_post_norm_projection.cc"
    norm_source = STATIC / "overlay/hc_post_norm_overlay.cc"
    projection_source = PROJECTION / "hc_projection_silu.cc"
    source.write_text(source_text(norm_source.read_text(), projection_source.read_text()))
    registry_source = OUTPUT / "webgpu_contrib_kernels.cc"
    registry_source.write_text(registry_text((REGISTRY / registry_source.name).read_text()))
    commands = []
    for template, path in ((component["compile_commands"][0], source),
                           (registry["compile_commands"][1], registry_source)):
        command = template.copy()
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
    comparator.write_text(replace_once((PROJECTION / comparator.name).read_text(),
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
        original_norm_source_sha256=sha(norm_source),
        original_projection_source_sha256=sha(projection_source),
        comparator_sha256=sha(OUTPUT / "native_compare"),
        no_runtime_object_layout_changes=True, compound_fallback_not_replay_cached=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
