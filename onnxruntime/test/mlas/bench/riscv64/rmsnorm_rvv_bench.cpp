/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rmsnorm_rvv_bench.cpp

Abstract:

    Correctness and performance comparison of RMSNorm (SimplifiedLayerNorm).

    Scalar path: ORT's ComputeJob<float> with simplified=true
      (anonymous namespace in layer_norm_impl.cc, reproduced here verbatim)
    MLAS path: MlasLayerNormF32 dispatch (uses RVV kernel when available)

--*/

#include "mlas.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

struct Options {
  size_t hidden = 1024;
  size_t iters = 500;
  size_t warmup = 50;
};

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    const auto split = arg.find('=');
    if (split == std::string_view::npos) continue;
    const auto key = arg.substr(0, split);
    const auto value = arg.substr(split + 1);
    if (key == "--hidden")
      options.hidden = std::strtoull(value.data(), nullptr, 10);
    else if (key == "--iters")
      options.iters = std::strtoull(value.data(), nullptr, 10);
    else if (key == "--warmup")
      options.warmup = std::strtoull(value.data(), nullptr, 10);
  }
  return options;
}

float MakeValue(size_t index) {
  uint32_t x = static_cast<uint32_t>(index * 747796405u + 2891336453u);
  x ^= x >> 16;
  x *= 2246822519u;
  x ^= x >> 13;
  return (static_cast<float>(x % 2048u) / 1024.0f) - 1.0f;
}

template <typename Fn>
double TimeLoop(size_t iterations, Fn&& fn) {
  const auto begin = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    fn();
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

//
// ORT scalar path: verbatim from layer_norm_impl.cc ComputeJob<float>
// with simplified=true (RMSNorm).
//
void OrtRmsNormScalar(
    const float* input,
    const float* scale,
    size_t norm_size,
    float epsilon,
    float* output) {
  float mean_square = 0.0f;
  for (size_t h = 0; h < norm_size; h++) {
    output[h] = input[h];
    mean_square += input[h] * input[h];
  }
  mean_square = sqrtf(mean_square / static_cast<float>(norm_size) + epsilon);
  for (size_t h = 0; h < norm_size; h++) {
    output[h] = output[h] / mean_square * scale[h];
  }
}

void OrtRmsNormMlas(
    const float* input,
    const float* scale,
    size_t norm_size,
    float epsilon,
    float* output) {
  if (!MlasLayerNormF32(input, scale, nullptr, output, nullptr, nullptr,
                        norm_size, epsilon, true)) {
    OrtRmsNormScalar(input, scale, norm_size, epsilon, output);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = ParseArgs(argc, argv);
  const size_t N = opts.hidden;
  const float epsilon = 1e-6f;

  std::cout << "=== RMSNorm: MLAS Dispatch vs ORT Scalar ===\n"
            << "  hidden=" << N << " iters=" << opts.iters << " warmup=" << opts.warmup << "\n\n";

  std::vector<float> input(N), scale(N);
  std::vector<float> out_scalar(N), out_rvv(N);

  for (size_t i = 0; i < N; i++) {
    input[i] = MakeValue(i) * 0.1f;
    scale[i] = 1.0f + MakeValue(i + N) * 0.01f;
  }

  // --- Correctness ---
  OrtRmsNormScalar(input.data(), scale.data(), N, epsilon, out_scalar.data());
  OrtRmsNormMlas(input.data(), scale.data(), N, epsilon, out_rvv.data());

  double max_abs = 0.0, max_rel = 0.0;
  size_t mismatches = 0;
  for (size_t i = 0; i < N; i++) {
    double abs_err = std::abs(out_scalar[i] - out_rvv[i]);
    double rel_err = (std::abs(out_scalar[i]) > 1e-7) ? abs_err / std::abs(out_scalar[i]) : abs_err;
    if (abs_err > max_abs) max_abs = abs_err;
    if (rel_err > max_rel) max_rel = rel_err;
    if (abs_err > 1e-5) mismatches++;
  }

  std::cout << "Correctness:\n"
            << "  max_abs=" << max_abs << " max_rel=" << max_rel
            << " mismatches=" << mismatches << "/" << N
            << (mismatches == 0 ? " PASS" : " FAIL") << "\n";

  // --- Performance ---
  auto run_scalar = [&]() {
    OrtRmsNormScalar(input.data(), scale.data(), N, epsilon, out_scalar.data());
  };
  auto run_rvv = [&]() {
    OrtRmsNormMlas(input.data(), scale.data(), N, epsilon, out_rvv.data());
  };

  for (size_t i = 0; i < opts.warmup; i++) {
    run_scalar();
    run_rvv();
  }

  double scalar_ms = TimeLoop(opts.iters, run_scalar) / opts.iters;
  double rvv_ms = TimeLoop(opts.iters, run_rvv) / opts.iters;

  std::cout << std::fixed << std::setprecision(4)
            << "\nPerformance:\n"
            << "  ORT Scalar: " << scalar_ms * 1000 << " us\n"
            << "  RVV:        " << rvv_ms * 1000 << " us\n"
            << "  Speedup:    " << scalar_ms / rvv_ms << "x\n";

  return (mismatches > 0) ? 1 : 0;
}
