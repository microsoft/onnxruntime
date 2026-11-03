import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


def program_source(original):
    body = original.split("class HCExactDenseProgram final", 1)[1].split(
        "\nclass MatMulNBitsSkinnyProgram", 1)[0]
    body = "class HCExactDenseProgram final" + body
    body = body.replace("HCExactDenseProgram", "HCProjectionSiLUProgram").replace(
        'Program{"HCExactDense"}', 'Program{"HCProjectionSiLUF16"}')
    body = replace_once(body,
        '    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);',
        '    const auto& activated = shader.AddOutput("activated", ShaderUsage::UseElementTypeAlias);\n'
        '    const auto& tail = shader.AddOutput("tail");\n'
        '    shader.AdditionalImplementation() <<\n'
        '        "fn stable_sigmoid(v: f32) -> f32 {\\n"\n'
        '        "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\\n"\n'
        '        "  let e = exp(v); return e / (1.0 + e);\\n}\\n";')
    body = replace_once(body, '         << y.SetByOffset("col", "value") << "\\n"',
        '''         << "  if (col < uniforms.prefix) {\\n"
         << "    let z_t = activated_element_t(f32(value) * uniforms.alpha);\\n"
         << "    let sigmoid_t = activated_element_t(stable_sigmoid(f32(z_t)));\\n"
         << activated.SetByOffset("col", "activated_element_t(f32(z_t) * f32(sigmoid_t))") << "\\n"
         << "  } else {\\n"
         << tail.SetByOffset("col - uniforms.prefix", "value") << "\\n"
         << "  }\\n"''')
    body = replace_once(body, "\n};", """
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"prefix", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32});
};""")
    return body


def main():
    root = Path(__file__).parent
    repos = Path("/workspace")
    prior_dir = repos / "qwen38-int4-static-node-replay/build-v8"
    prior = json.loads((prior_dir / "build-manifest.json").read_text())
    output = repos / "qwen38-int4-hc-projection-silu/build-v1"
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
    original = (repos / "onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4/matmul_nbits.cc").read_text()
    source = output / "hc_projection_silu.cc"
    source.write_text('''#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/bert/scaled_silu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "static_node_replay.h"

namespace onnxruntime::contrib::webgpu {
''' + program_source(original) + "\n" +
        (root / "hc_projection_silu_kernel.inc").read_text() + "\n}\n")
    registry = output / "webgpu_contrib_kernels.cc"
    registry_text = (repos / "qwen38-int4-hc-post-norm/build-v8/webgpu_contrib_kernels.cc").read_text()
    anchor = "    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, HCSharedPostNormF16)>,"
    registry.write_text(replace_once(registry_text, anchor,
        anchor + "\n" + anchor.replace("HCSharedPostNormF16", "HCProjectionSiLUF16")))
    commands = []
    for path in (source, registry):
        obj = output / (path.name + ".o")
        command = prior["compile_commands"][0].copy()
        command[1:1] = ["-I" + str(prior_dir / "overlay")]
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
    text = replace_once(text, 'setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1)',
                        'setenv("ORT_WEBGPU_HC_PROJECTION_SILU", value, 1)')
    text = replace_once(text, "if (i == 0) width = shape.back();", "if (i == 0) width = 324;")
    comparator.write_text(text)
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(output / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
        compile_commands=commands, link_command=link, working_directory=prior["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, registry, comparator)},
        original_exact_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
