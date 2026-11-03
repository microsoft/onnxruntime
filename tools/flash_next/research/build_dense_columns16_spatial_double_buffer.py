import json
from pathlib import Path
import subprocess

from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from build_dense_columns8_word_unroll import PROGRAM as REPLACED_PROGRAM, transform as selector_transform, SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_COLUMNS16_SPATIAL_DOUBLE_BUFFER"
CLASS = "DenseColumns16SpatialDoubleBufferProgram"
OUTPUT = Path("/workspace/qwen38-int4-dense-columns16-spatial-double-buffer/build-v1")


def make_program():
    source = ORIGINAL_PROGRAM.replace("DenseColumns8DoubleBuffer", "DenseColumns16SpatialDoubleBuffer")
    source = replace_once(source, "partials: array<f16, 256>", "partials: array<f16, 512>")
    source = replace_once(source,
        "for (var i = local_idx; i < 256u; i += 128u) { partials[i] = f16(0); }",
        "for (var i = local_idx; i < 512u; i += 512u) { partials[i] = f16(0); }")
    source = replace_once(source, "for (var i = local_idx; i < 256u; i += 128u)",
                          "for (var i = local_idx; i < 256u; i += 512u)")
    start = source.index('    code << "    let col1 =')
    stop = source.index('    code << R"(', start)
    source = source[:start] + source[stop:]
    source = replace_once(source, """    code << "      {\\n"
         << "        let low = decode_low_nibbles(packed1[word]);\\n"
         << "        let high = decode_low_nibbles(packed1[word] >> 4u);\\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale1;\\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale1;\\n"
         << "        sum1 += dot(x0, w0) + dot(x1, w1);\\n"
         << "      }\\n";
""", "")
    source = replace_once(source,
        '    code << "    partials[(column_group + 4u) * 32u + stream] += sum1;\\n";\n', "")
    source = replace_once(source, "let col0 = workgroup_idx * 8u", "let col0 = workgroup_idx * 16u")
    source = replace_once(source, "if (local_idx < 8u)", "if (local_idx < 16u)")
    return replace_once(source, 'y.SetByOffset("workgroup_idx * 8u + local_idx"',
                        'y.SetByOffset("workgroup_idx * 16u + local_idx"')


PROGRAM = make_program()

TIMING_BODY = r'''
    if (timing) {
      size_t consumed = 0;
      const int iterations = std::stoi(argv[5], &consumed);
      if (consumed != std::strlen(argv[5]) || iterations < 128 || iterations > 20000)
        throw std::runtime_error("Timing iterations must be an integer in [128, 20000]");
      for (size_t i = 0; i < k; ++i)
        input[i] = Ort::Float16_t(0.125f * std::sin(float(i + 1) * 0.071f));
      auto value = Ort::Value::CreateTensor<Ort::Float16_t>(memory, input.data(), input.size(),
                                                         shape.data(), shape.size());
      std::vector<double> durations;
      durations.reserve(iterations);
      for (int iteration = -8; iteration < iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        auto actual = session.Run(Ort::RunOptions{}, names, &value, 1, outputs, 1);
        const auto end = std::chrono::steady_clock::now();
        const auto info = actual[0].GetTensorTypeAndShapeInfo();
        if (info.GetShape() != output_shape || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Unexpected timing output");
        const auto* observed = actual[0].GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < n; ++i) {
          if (!std::isfinite(reference[i].ToFloat()) || !std::isfinite(observed[i].ToFloat()) ||
              std::memcmp(&reference[i], &observed[i], sizeof(reference[i])) != 0)
            throw std::runtime_error("Timed output differs from finite independent reference");
        }
        if (iteration >= 0)
          durations.push_back(std::chrono::duration<double, std::micro>(end - start).count());
      }
      double total = 0;
      for (const auto duration : durations) total += duration;
      std::sort(durations.begin(), durations.end());
      std::cout << "TIMING K=" << k << " N=" << n << " iterations=" << iterations
                << " warmup=8 comparisons=" << n * iterations << " mismatches=0 mean_us="
                << total / iterations << " median_us="
                << (durations[(iterations - 1) / 2] + durations[iterations / 2]) / 2
                << " profiling=0" << std::endl;
      return 0;
    }
'''


def timing_comparator(source):
    source = replace_once(source, "#include <array>", "#include <algorithm>\n#include <chrono>\n#include <array>")
    source = replace_once(source, 'if (mode != "capture" && mode != "compare")',
                          'if (mode != "capture" && mode != "compare" && mode != "timing")')
    source = replace_once(source, '    const bool capture = mode == "capture";',
                          '    const bool capture = mode == "capture";\n'
                          '    const bool timing = mode == "timing";')
    source = replace_once(source, "    options.EnableProfiling(argv[5]);",
                          "    if (!timing) options.EnableProfiling(argv[5]);")
    return replace_once(source, '    const char* outputs[]{"y"};\n',
                        '    const char* outputs[]{"y"};\n' + TIMING_BODY)


def transform(source):
    result = replace_once(selector_transform(source), REPLACED_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_DENSE_COLUMNS8_WORD_UNROLL", FLAG)
    result = result.replace("word_unroll", "spatial16")
    result = replace_once(result, "DenseColumns8WordUnrollProgram program;", CLASS + " program;")
    result = replace_once(result, "  auto compute = [&]() {",
        "  const bool use_spatial16 = spatial16 == 1 && N_ % 16 == 0 &&\n"
        '      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&\n'
        "      context.DeviceLimits().maxComputeWorkgroupSizeX >= 512 &&\n"
        "      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 512 &&\n"
        "      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 5120;\n"
        "  auto compute = [&]() {")
    result = replace_once(result, "if (columns8 && spatial16 == 1)", "if (columns8 && use_spatial16)")
    start = result.index("    if (columns8 && use_spatial16)")
    end = result.index("\n    }", start)
    selected = result[start:end]
    selected = replace_once(selected, ".SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 8);",
                            ".SetWorkgroupSize(512).SetDispatchGroupSize(N_ / 16);")
    result = result[:start] + selected + result[end:]
    return replace_once(result, "columns8 ? (spatial16 == 1 ? 8 : 7)",
                        "columns8 ? (use_spatial16 ? 8 : 7)")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, output_columns=16, physical_threads=512,
             shared_storage_bytes=5120, activation_buffers=2, columns_per_thread=1,
             spatial_column_concurrency=True, supported_double_buffer_selector_preserved=True,
             original_dot_word_tile_and_shared_partial_order_preserved=True,
             held_private_totals_word_unroll_and_swizzle_not_stacked=True,
             original_provider_full_kernel_comparison_required=True))
    cpp = OUTPUT / "paired_matrix_compare.cpp"
    cpp.write_text((ROOT / "dense_paired_matrix_compare.cpp").read_text())
    binary = OUTPUT / "paired_matrix_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(cpp),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(binary)], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  paired_comparator_sha256=sha(binary))
    report["source_sha256"]["paired_matrix_compare.cpp"] = sha(cpp)
    timing_cpp = OUTPUT / "unprofiled_compare.cpp"
    timing_cpp.write_text(timing_comparator(cpp.read_text()))
    timing_binary = OUTPUT / "unprofiled_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(timing_cpp),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(timing_binary)], check=True)
    report["source_sha256"]["unprofiled_compare.cpp"] = sha(timing_cpp)
    report["unprofiled_comparator_sha256"] = sha(timing_binary)
    report["unprofiled_timing_includes_host_run_and_readback"] = True
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
