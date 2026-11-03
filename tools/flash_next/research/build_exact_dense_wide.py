import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent
    prior_dir = Path("/workspace/qwen38-int4-quickgelu-next/build")
    prior = json.loads((prior_dir / "build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-exact-wide-gpu7" +
                  ("-filtered" if args.filtered else "") + "/build")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    link = prior["link_command"].copy()
    baseline = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(baseline)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    assert sha(baseline) == prior["provider_sha256"], "Baseline relink mismatch"
    baseline.unlink()
    retained = Path("/workspace/onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4/matmul_nbits.cc")
    text = retained.read_text()
    assert text.count("class ExactDenseMidProgram final") == 1
    text = text.replace("class ExactDenseMidProgram final",
                        (root / "exact_dense_wide_program.inc").read_text() + "\nclass ExactDenseMidProgram final", 1)
    anchor = '  // Adapt the few-row, K-parallel GEMV strategy'
    assert text.count(anchor) == 1
    gate = """  if (ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_WIDE", 0) != 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && M == 1 && dispatch_M == M && K >= 32 && K % 32 == 0 &&
      N > 0 && (N + 1) / 2 <= context.DeviceLimits().maxComputeWorkgroupsPerDimension &&
      nbits == 4 && block_size == 32 && !has_zero_points && !has_bias && !has_weight_idx &&
      !context.EnableMatmulFp32Accumulation() &&
      a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
      scales->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
    ExactDenseWideProgram program{K / 32, N};
    program.SetWorkgroupSize(64).SetDispatchGroupSize((N + 1) / 2);
    program.CacheHint(K, N)
        .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
    return context.RunProgram(program);
  }

"""
    if args.filtered:
        gate = gate.replace("K >= 32 && K % 32 == 0 &&",
                            "K >= 32 && K % 32 == 0 && !(K == 2560 && N > 1) &&")
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
    native = native.replace("ORT_WEBGPU_HC_EXACT_DENSE", "ORT_WEBGPU_EXACT_DENSE_WIDE")
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
    report = dict(status="built_not_qualified", provider_sha256=sha(baseline),
                  baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
                  compile_command=compile_command, link_command=link,
                  working_directory=prior["working_directory"], native_compile_command=native_command,
                  source_sha256={str(p): sha(p) for p in (source, native_source)},
                  filtered=args.filtered,
                  program_generator_sha256=sha(root / "exact_dense_wide_program.inc"),
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
