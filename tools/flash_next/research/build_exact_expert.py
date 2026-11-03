import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).parent
    prior_dir = Path("/workspace/qwen38-int4-quickgelu-next/build")
    prior = json.loads((prior_dir / "build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-exact-expert/build-v2")
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
    retained = Path("/workspace/onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4/matmul_nbits.cc")
    text = retained.read_text()
    assert text.count("class ExactDenseMidProgram final") == 1
    text = text.replace("class ExactDenseMidProgram final",
                        (root / "exact_expert_program.inc").read_text() + "\nclass ExactDenseMidProgram final", 1)
    anchor = '  // Adapt the few-row, K-parallel GEMV strategy'
    assert text.count(anchor) == 1
    gate = """  const int expert_columns = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_EXPERT", 0);
  ORT_RETURN_IF_NOT(expert_columns == 0 || expert_columns == 2 || expert_columns == 4 || expert_columns == 8,
                    "ORT_WEBGPU_EXACT_EXPERT must be 0, 2, 4 or 8.");
  if (expert_columns != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && dispatch_M <= 16 && (M == 1 || M == dispatch_M) &&
      K >= 32 && K % 32 == 0 && N > 0 &&
      has_weight_idx_indirect && nbits == 4 && block_size == 32 &&
      !has_zero_points && !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    const uint32_t columns_per_group = static_cast<uint32_t>(expert_columns);
    const uint32_t groups = (N + columns_per_group - 1) / columns_per_group * dispatch_M;
    ORT_RETURN_IF_NOT(groups <= context.DeviceLimits().maxComputeWorkgroupsPerDimension,
                      "Exact expert dispatch exceeds per-dimension workgroup limits.");
    ExactExpertProgram program{K / 32, N, columns_per_group, M == 1, has_bias};
    program.SetWorkgroupSize(columns_per_group * 32).SetDispatchGroupSize(groups);
    program.CacheHint(K, N, columns_per_group, M == 1, has_bias)
        .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_bias) program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

"""
    source = output / "matmul_nbits.cc"
    source.write_text(text.replace(anchor, gate + anchor, 1))
    obj = output / "matmul_nbits.cc.o"
    compile_command = prior["compile_commands"][0].copy()
    compile_command[compile_command.index("-c") + 1] = str(source)
    compile_command[compile_command.index("-o") + 1] = str(obj)
    subprocess.run(compile_command, cwd=prior["working_directory"], env=environment, check=True)
    old = "/workspace/onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4/matmul_nbits.cc.o"
    assert link.count(old) == 1
    link = [str(obj) if value == old else value for value in link]
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    native = Path("/workspace/qwen38-int4-dense-next/native_compare.cpp").read_text()
    native = native.replace("ORT_WEBGPU_HC_EXACT_DENSE", "ORT_WEBGPU_EXACT_EXPERT")
    native = native.replace('mode ? "1" : "0"', 'mode ? std::getenv("EXPERT_COLUMNS") : "0"')
    native = native.replace("round < 42", "round < 8").replace("times[0][20]", "times[0][3]").replace(
        "times[1][20]", "times[1][3]").replace("samples_per_mode=40", "samples_per_mode=6")
    native_source = output / "native_compare.cpp"
    native_source.write_text(native)
    native_command = ["c++", "-O3", "-std=c++20", str(native_source),
                      "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                      "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                      "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                      "-lonnxruntime", "-o", str(output / "native_compare")]
    subprocess.run(native_command, env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
                  compile_command=compile_command, link_command=link,
                  working_directory=prior["working_directory"], native_compile_command=native_command,
                  source_sha256={str(p): sha(p) for p in (source, native_source)},
                  program_generator_sha256=sha(root / "exact_expert_program.inc"),
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
