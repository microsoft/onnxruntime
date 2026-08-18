/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rope_rvv_bench.cpp

Abstract:

    Correctness and performance comparison of RotaryEmbedding.

    Scalar path: MlasRotaryEmbedOneRow_FallBack (ORT's internal scalar fallback)
    Dispatch path: MlasRotaryEmbedOneRow (dispatches to RVV kernel via platform)

--*/

#include "mlas.h"
#include "mlas_float16.h"
#include "rotary_embedding.h"

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
  size_t dim = 128;
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
    if (key == "--dim")
      options.dim = std::strtoull(value.data(), nullptr, 10);
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

void CompareResults(const float* ref, const float* got, size_t n) {
  double max_abs = 0.0, max_rel = 0.0;
  size_t mismatches = 0;
  for (size_t i = 0; i < n; i++) {
    double abs_err = std::abs(ref[i] - got[i]);
    double rel_err = (std::abs(ref[i]) > 1e-7) ? abs_err / std::abs(ref[i]) : abs_err;
    if (abs_err > max_abs) max_abs = abs_err;
    if (rel_err > max_rel) max_rel = rel_err;
    if (abs_err > 1e-5) mismatches++;
  }
  std::cout << "  max_abs=" << max_abs << " max_rel=" << max_rel
            << " mismatches=" << mismatches << "/" << n
            << (mismatches == 0 ? " PASS" : " FAIL") << "\n";
}

void BenchRoPE(const char* label, size_t dim, bool interleaved, size_t iters, size_t warmup) {
  if (dim % 2 != 0) {
    std::cerr << "Error: dim must be even, got " << dim << "\n";
    return;
  }
  const size_t half = dim / 2;

  std::vector<float> input(dim), sin_data(half), cos_data(half);
  std::vector<float> out_fallback(dim), out_dispatch(dim);

  for (size_t i = 0; i < dim; i++) input[i] = MakeValue(i);
  for (size_t i = 0; i < half; i++) {
    sin_data[i] = sinf(static_cast<float>(i) * 0.01f);
    cos_data[i] = cosf(static_cast<float>(i) * 0.01f);
  }

  // ORT scalar fallback
  MlasRotaryEmbedOneRow_FallBack<float>(
      input.data(), sin_data.data(), cos_data.data(), dim, interleaved, out_fallback.data());
  // ORT dispatch (→ RVV)
  MlasRotaryEmbedOneRow<float>(
      input.data(), sin_data.data(), cos_data.data(), dim, interleaved, out_dispatch.data());

  std::cout << "--- " << label << " (dim=" << dim << ") ---\n";
  CompareResults(out_fallback.data(), out_dispatch.data(), dim);

  auto run_fallback = [&]() {
    MlasRotaryEmbedOneRow_FallBack<float>(
        input.data(), sin_data.data(), cos_data.data(), dim, interleaved, out_fallback.data());
  };
  auto run_dispatch = [&]() {
    MlasRotaryEmbedOneRow<float>(
        input.data(), sin_data.data(), cos_data.data(), dim, interleaved, out_dispatch.data());
  };

  for (size_t i = 0; i < warmup; i++) {
    run_fallback();
    run_dispatch();
  }

  double fallback_ms = TimeLoop(iters, run_fallback) / iters;
  double dispatch_ms = TimeLoop(iters, run_dispatch) / iters;

  std::cout << std::fixed << std::setprecision(4)
            << "  ORT Fallback: " << fallback_ms * 1000 << " us\n"
            << "  ORT Dispatch: " << dispatch_ms * 1000 << " us\n"
            << "  Speedup:      " << fallback_ms / dispatch_ms << "x\n\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = ParseArgs(argc, argv);

  std::cout << "=== RotaryEmbedding: RVV Dispatch vs ORT Scalar Fallback ===\n\n";

  BenchRoPE("RoPE non-interleaved", opts.dim, false, opts.iters, opts.warmup);
  BenchRoPE("RoPE interleaved", opts.dim, true, opts.iters, opts.warmup);

  return 0;
}
