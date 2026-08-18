/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_rvv_bench.cpp

Abstract:

    Correctness and performance comparison of RVV-accelerated FP16 GEMM
    against ORT's built-in scalar FP16 GEMM dispatch (MlasHalfGemmDispatchDefault).

    Both paths use the same MLAS HalfGemm dispatch interface with FP16 I/O.

    Usage:
      ./onnxruntime_mlas_halfgemm_rvv_bench [--m=N] [--n=N] [--k=N]
          [--iters=N] [--warmup=N] [--bias=0|1]

--*/

#include "mlas.h"
#include "mlas_float16.h"
#include "halfgemm.h"

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
  size_t m = 64;
  size_t n = 768;
  size_t k = 768;
  size_t iters = 20;
  size_t warmup = 3;
  bool use_bias = false;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--m=N] [--n=N] [--k=N] [--iters=N] [--warmup=N] [--bias=0|1]\n";
}

bool ParseBool(std::string_view value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    const auto split = arg.find('=');
    if (split == std::string_view::npos || split == 0 || split + 1 >= arg.size()) {
      continue;
    }
    const std::string_view key = arg.substr(0, split);
    const std::string_view value = arg.substr(split + 1);
    if (key == "--m")
      options.m = std::strtoull(value.data(), nullptr, 10);
    else if (key == "--n")
      options.n = std::strtoull(value.data(), nullptr, 10);
    else if (key == "--k")
      options.k = std::strtoull(value.data(), nullptr, 10);
    else if (key == "--iters")
      options.iters = std::strtoull(value.data(), nullptr, 10);
    else if (key == "--warmup")
      options.warmup = std::strtoull(value.data(), nullptr, 10);
    else if (key == "--bias")
      options.use_bias = ParseBool(value);
  }
  return options;
}

float MakeValue(size_t index) {
  uint32_t x = static_cast<uint32_t>(index * 747796405u + 2891336453u);
  x ^= x >> 16;
  x *= 2246822519u;
  x ^= x >> 13;
  const uint32_t bucket = x % 2048u;
  return (static_cast<float>(bucket) / 1024.0f) - 1.0f;
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

void RunDispatch(
    const MLAS_HALFGEMM_DISPATCH& dispatch,
    size_t M, size_t N, size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* data) {
  dispatch.Operation(N, K, data, 0, M, 0, N);
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = ParseArgs(argc, argv);

  if (options.m == 0 || options.n == 0 || options.k == 0 || options.iters == 0) {
    std::cerr << "m, n, k, and iters must be > 0\n";
    return 1;
  }

  const bool fp16_supported = MlasFp16AccelerationSupported();

  std::cout << "=== FP16 GEMM: RVV vs ORT Scalar Dispatch ===\n"
            << "  M=" << options.m << " N=" << options.n << " K=" << options.k
            << " bias=" << (options.use_bias ? "yes" : "no") << "\n"
            << "  iters=" << options.iters << " warmup=" << options.warmup << "\n"
            << "  FP16 acceleration: " << (fp16_supported ? "YES (RVV)" : "NO") << "\n\n";

  const size_t a_size = options.m * options.k;
  const size_t b_size = options.k * options.n;
  const size_t c_size = options.m * options.n;

  std::vector<_mlas_fp16_> a_fp16(a_size);
  std::vector<_mlas_fp16_> b_fp16(b_size);
  std::vector<_mlas_fp16_> bias_fp16(options.n);
  std::vector<_mlas_fp16_> c_rvv(c_size);
  std::vector<_mlas_fp16_> c_scalar(c_size);

  for (size_t i = 0; i < a_size; ++i) {
    a_fp16[i] = MLAS_Float2Half(MakeValue(i) * 0.1f);
  }
  for (size_t i = 0; i < b_size; ++i) {
    b_fp16[i] = MLAS_Float2Half(MakeValue(i + a_size) * 0.1f);
  }
  if (options.use_bias) {
    for (size_t i = 0; i < options.n; ++i) {
      bias_fp16[i] = MLAS_Float2Half(MakeValue(i + a_size + b_size) * 0.01f);
    }
  }

  MLAS_HALF_GEMM_DATA_PARAMS params_scalar;
  params_scalar.A = a_fp16.data();
  params_scalar.lda = options.k;
  params_scalar.B = b_fp16.data();
  params_scalar.ldb = options.n;
  params_scalar.C = reinterpret_cast<MLAS_FP16*>(c_scalar.data());
  params_scalar.ldc = options.n;
  params_scalar.Bias = options.use_bias
                           ? reinterpret_cast<const MLAS_FP16*>(bias_fp16.data())
                           : nullptr;
  params_scalar.AIsfp32 = false;
  params_scalar.BIsfp32 = false;
  params_scalar.OutputProcessor = nullptr;

  MLAS_HALF_GEMM_DATA_PARAMS params_rvv = params_scalar;
  params_rvv.C = reinterpret_cast<MLAS_FP16*>(c_rvv.data());

  // --- Run both dispatches ---
  RunDispatch(MlasHalfGemmDispatchDefault, options.m, options.n, options.k, &params_scalar);

#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV_ZVFH)
  RunDispatch(MlasHalfGemmDispatchRvv, options.m, options.n, options.k, &params_rvv);
#else
  RunDispatch(MlasHalfGemmDispatchDefault, options.m, options.n, options.k, &params_rvv);
  std::cout << "  (RVV dispatch not available, comparing scalar vs scalar)\n\n";
#endif

  // --- Correctness: RVV vs ORT Scalar ---
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  size_t error_count = 0;

  for (size_t i = 0; i < c_size; ++i) {
    float ref = MLAS_Half2Float(c_scalar[i]);
    float got = MLAS_Half2Float(c_rvv[i]);
    double abs_err = std::abs(ref - got);
    double rel_err = (std::abs(ref) > 1e-6) ? abs_err / std::abs(ref) : abs_err;

    if (abs_err > max_abs_err) max_abs_err = abs_err;
    if (rel_err > max_rel_err) max_rel_err = rel_err;

    if (rel_err > 0.10 && abs_err > 0.005) {
      if (error_count < 10) {
        std::cerr << "  MISMATCH [" << i / options.n << "," << i % options.n
                  << "]: scalar=" << ref << " rvv=" << got
                  << " abs=" << abs_err << " rel=" << rel_err << "\n";
      }
      error_count++;
    }
  }

  std::cout << "Correctness (RVV vs ORT Scalar):\n"
            << "  max abs error: " << max_abs_err << "\n"
            << "  max rel error: " << max_rel_err << "\n"
            << "  mismatches (>10% rel && >0.005 abs): " << error_count
            << " / " << c_size << "\n";

  if (error_count > 0) {
    std::cout << "  STATUS: FAIL\n\n";
  } else {
    std::cout << "  STATUS: PASS\n\n";
  }

  // --- Performance ---
  auto run_scalar_fn = [&]() {
    RunDispatch(MlasHalfGemmDispatchDefault, options.m, options.n, options.k, &params_scalar);
  };

  auto run_rvv_fn = [&]() {
#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV_ZVFH)
    RunDispatch(MlasHalfGemmDispatchRvv, options.m, options.n, options.k, &params_rvv);
#else
    RunDispatch(MlasHalfGemmDispatchDefault, options.m, options.n, options.k, &params_rvv);
#endif
  };

  for (size_t i = 0; i < options.warmup; ++i) {
    run_scalar_fn();
    run_rvv_fn();
  }

  const double scalar_ms = TimeLoop(options.iters, run_scalar_fn);
  const double scalar_avg = scalar_ms / static_cast<double>(options.iters);

  const double rvv_ms = TimeLoop(options.iters, run_rvv_fn);
  const double rvv_avg = rvv_ms / static_cast<double>(options.iters);

  const double flops = 2.0 * options.m * options.n * options.k;
  const double scalar_gflops = flops / (scalar_avg * 1e6);
  const double rvv_gflops = flops / (rvv_avg * 1e6);
  const double speedup = scalar_avg / rvv_avg;

  std::cout << std::fixed << std::setprecision(3)
            << "Performance:\n"
            << "  ORT Scalar:  " << scalar_avg << " ms  (" << scalar_gflops << " GFLOPS)\n"
            << "  RVV:         " << rvv_avg << " ms  (" << rvv_gflops << " GFLOPS)\n"
            << "  Speedup:     " << speedup << "x\n";

  return (error_count > 0) ? 1 : 0;
}
