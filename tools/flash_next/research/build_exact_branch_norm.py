import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).parent
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    prior = json.loads(Path("/workspace/qwen38-int4-quickgelu-next/build/build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-exact-branch-norm/build-v2")
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
    original = repo / "onnxruntime/contrib_ops/webgpu/bert/branchwise_rms_norm.cc"
    text = original.read_text()
    anchor = "BranchwiseRMSNorm::BranchwiseRMSNorm(const OpKernelInfo& info)"
    assert text.count(anchor) == 1
    text = text.replace('#include "contrib_ops/cpu/hyper_connection_helper.h"',
                        '#include "contrib_ops/cpu/hyper_connection_helper.h"\n'
                        '#include "core/platform/env_var_utils.h"\n'
                        '#include <string_view>')
    text = text.replace(anchor, (root / "exact_branch_norm_program.inc").read_text() + "\n" + anchor)
    anchor = "  BranchwiseRMSNormProgram program{scale != nullptr, shared_scale};"
    assert text.count(anchor) == 1
    gate = """  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_BRANCH_NORM", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      context.HasFeature(wgpu::FeatureName::Subgroups) &&
      context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32 &&
      x->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    ExactBranchNormProgram program{scale != nullptr, shared_scale};
    program.CacheHint(scale != nullptr, shared_scale)
        .AddInput({x, ProgramTensorMetadataDependency::Type});
    if (scale != nullptr) program.AddInput({scale, ProgramTensorMetadataDependency::Type});
    program.AddOutput({y, ProgramTensorMetadataDependency::None})
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(params.hidden)},
                              {onnxruntime::narrow<uint32_t>(params.branches)}, {epsilon_}})
        .SetDispatchGroupSize(groups).SetWorkgroupSize(256);
    return context.RunProgram(program);
  }

"""
    source = output / "branchwise_rms_norm.cc"
    source.write_text(text.replace(anchor, gate + anchor))
    obj = output / "branchwise_rms_norm.cc.o"
    command = prior["compile_commands"][0].copy()
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=prior["working_directory"], env=environment, check=True)
    matches = [p for p in link if p.endswith("contrib_ops__webgpu__bert__branchwise_rms_norm.cc.o")]
    assert len(matches) == 1, matches
    link[link.index(matches[0])] = str(obj)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    native = Path("/workspace/qwen38-int4-dense-next/native_compare.cpp").read_text()
    native = native.replace("ORT_WEBGPU_HC_EXACT_DENSE", "ORT_WEBGPU_EXACT_BRANCH_NORM")
    native = native.replace("round < 42", "round < 8").replace("times[0][20]", "times[0][3]").replace(
        "times[1][20]", "times[1][3]").replace("samples_per_mode=40", "samples_per_mode=6")
    native_source = output / "native_compare.cpp"
    native_source.write_text(native)
    subprocess.run(["c++", "-O3", "-std=c++20", str(native_source),
                    "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                    "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-lonnxruntime", "-o", str(output / "native_compare")], check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
                  compile_command=command, link_command=link, working_directory=prior["working_directory"],
                  source_sha256={str(source): sha(source)}, quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
