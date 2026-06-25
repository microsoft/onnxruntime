/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast_rvv_bench.cpp

Abstract:

    Correctness and performance comparison of FP16<->FP32 cast kernels.

    Scalar path: ORT's internal fallback in cast.cpp
      (MLAS_Half2Float / MLAS_Float2Half loop when CastKernel == nullptr)
    Dispatch path: MlasConvertHalfToFloatBuffer / MlasConvertFloatToHalfBuffer
      (dispatches to registered RVV kernel via platform.CastF16ToF32Kernel)

--*/

#include "mlas.h"
#include "mlas_float16.h"

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
  size_t count = 1024 * 64;
  size_t iters = 200;
  size_t warmup = 20;
};

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    const auto split = arg.find('=');
    if (split == std::string_view::npos) continue;
    const auto key = arg.substr(0, split);
    const auto value = arg.substr(split + 1);
    if (key == "--count")
      options.count = std::strtoull(value.data(), nullptr, 10);
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

}  // namespace

int main(int argc, char** argv) {
  const Options opts = ParseArgs(argc, argv);
  const size_t N = opts.count;

  std::cout << "=== FP16<->FP32 Cast: RVV Dispatch vs ORT Scalar Fallback ===\n"
            << "  count=" << N << " iters=" << opts.iters << " warmup=" << opts.warmup << "\n\n";

  std::vector<float> fp32_src(N);
  std::vector<_mlas_fp16_> fp16_src(N);
  for (size_t i = 0; i < N; ++i) {
    fp32_src[i] = MakeValue(i);
    fp16_src[i] = MLAS_Float2Half(fp32_src[i]);
  }

  std::vector<float> f16_to_f32_fallback(N), f16_to_f32_dispatch(N);
  std::vector<_mlas_fp16_> f32_to_f16_fallback(N), f32_to_f16_dispatch(N);

  // ORT scalar fallback: same as cast.cpp when CastF16ToF32Kernel == nullptr
  //   for (i) Destination[i] = Source[i].ToFloat();  // calls MLAS_Half2Float
  auto fallback_h2f = [&]() {
    for (size_t i = 0; i < N; ++i)
      f16_to_f32_fallback[i] = MLAS_Half2Float(fp16_src[i]);
  };
  auto fallback_f2h = [&]() {
    for (size_t i = 0; i < N; ++i)
      f32_to_f16_fallback[i] = MLAS_Float2Half(fp32_src[i]);
  };

  // ORT dispatch path: MlasConvertHalfToFloatBuffer (uses registered RVV kernel)
  auto dispatch_h2f = [&]() {
    MlasConvertHalfToFloatBuffer(
        reinterpret_cast<const MLAS_FP16*>(fp16_src.data()),
        f16_to_f32_dispatch.data(), N);
  };
  auto dispatch_f2h = [&]() {
    MlasConvertFloatToHalfBuffer(
        fp32_src.data(),
        reinterpret_cast<MLAS_FP16*>(f32_to_f16_dispatch.data()), N);
  };

  // --- Correctness ---
  fallback_h2f();
  dispatch_h2f();
  fallback_f2h();
  dispatch_f2h();

  size_t h2f_mismatches = 0;
  for (size_t i = 0; i < N; ++i) {
    if (f16_to_f32_fallback[i] != f16_to_f32_dispatch[i]) h2f_mismatches++;
  }
  size_t f2h_mismatches = 0;
  for (size_t i = 0; i < N; ++i) {
    if (f32_to_f16_fallback[i] != f32_to_f16_dispatch[i]) f2h_mismatches++;
  }

  std::cout << "Correctness:\n"
            << "  F16->F32: mismatches=" << h2f_mismatches << "/" << N
            << (h2f_mismatches == 0 ? " PASS" : " FAIL") << "\n"
            << "  F32->F16: mismatches=" << f2h_mismatches << "/" << N
            << (f2h_mismatches == 0 ? " PASS" : " FAIL") << "\n";

  // --- Performance ---
  for (size_t i = 0; i < opts.warmup; ++i) {
    fallback_h2f();
    dispatch_h2f();
    fallback_f2h();
    dispatch_f2h();
  }

  double s_h2f = TimeLoop(opts.iters, fallback_h2f) / opts.iters;
  double d_h2f = TimeLoop(opts.iters, dispatch_h2f) / opts.iters;
  double s_f2h = TimeLoop(opts.iters, fallback_f2h) / opts.iters;
  double d_f2h = TimeLoop(opts.iters, dispatch_f2h) / opts.iters;

  std::cout << std::fixed << std::setprecision(3)
            << "\nF16->F32 (" << N << " elements):\n"
            << "  ORT Fallback: " << s_h2f << " ms\n"
            << "  ORT Dispatch: " << d_h2f << " ms\n"
            << "  Speedup:      " << s_h2f / d_h2f << "x\n"
            << "\nF32->F16 (" << N << " elements):\n"
            << "  ORT Fallback: " << s_f2h << " ms\n"
            << "  ORT Dispatch: " << d_f2h << " ms\n"
            << "  Speedup:      " << s_f2h / d_f2h << "x\n";

  return (h2f_mismatches + f2h_mismatches > 0) ? 1 : 0;
}
