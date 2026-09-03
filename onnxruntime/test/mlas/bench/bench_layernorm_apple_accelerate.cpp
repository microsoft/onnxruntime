// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// A/B benchmark: the Apple Accelerate (vDSP) LayerNorm/RMSNorm kernel
// (test_layernorm_apple_accelerate.cpp's subject) versus the portable
// scalar baseline it replaces on macOS arm64. Unlike an earlier, unrelated
// Apple Accelerate benchmark -- which compared vForce against an ALREADY
// NEON-vectorized polynomial kernel (see the closed, unmerged PR #32036,
// which found vForce consistently slower there) -- LayerNorm/RMSNorm has NO
// SIMD-vectorized kernel on ARM64 today: GetMlasPlatform().LayerNormF32Kernel
// is only registered for RVV (RISC-V); see core/mlas/lib/platform.cpp. Every
// ARM64 build without this option falls back to the single-element-at-a-time
// Welford/sum-of-squares scalar loop in
// onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc's ComputeJob --
// reproduced here as BM_LayerNormPortableScalarBaseline so the comparison
// reflects the REAL choice being made on ARM64, not a hypothetical one.
//
// Representative sizes below match common transformer hidden dims (BERT
// 768, GPT-2 768/1024/1280/1600, Phi-3/Gemma 3072, Llama-2/3 4096-8192,
// GPT-3-class 12288/16384), plus a few small sizes to observe any crossover
// where per-call vDSP/function-call overhead might outweigh its reduction
// speedup for very short rows -- the same question #32036 answered
// (negatively, for Tanh) for its own kernel; this benchmark exists to
// answer it for LayerNorm with real hardware data rather than assuming an
// answer. Sizes above kApplePerRowStackScratch (8192, see layernorm.cpp)
// exercise the heap-fallback scratch path, which has a different allocation
// cost profile than the on-stack path used at and below that threshold.
//
// The BM_*Dispatch benchmarks call the public MlasLayerNormF32 dispatch
// entry point directly. MlasLayerNormF32 returns false and does nothing when
// no LayerNormF32Kernel is registered or when the Apple kernel declines a
// row below its measured 64-element crossover -- it does NOT run the scalar
// fallback itself (only real ORT call sites do that, one layer up, in
// layer_norm_impl.cc). RunLayerNormBenchmark detects both cases and skips
// the dispatch benchmark rather than reporting a fake, near-zero-cost
// throughput number for a no-op.
//
// No Apple Silicon hardware was available in the environment that authored
// this benchmark; see the PR description for the actual measured numbers
// collected on macOS arm64 CI.

#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

#include "mlas.h"
#include "bench_util.h"
#include "core/mlas/lib/mlasi.h"

namespace {

// Reproduces onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc's
// ComputeJob scalar fallback exactly (Welford's online algorithm for full
// LayerNorm; single-pass sum-of-squares for RMSNorm/Simplified), so this
// benchmark measures the vDSP kernel against the REAL alternative it
// replaces on ARM64 builds without onnxruntime_USE_APPLE_ACCELERATE, not an
// approximation of it.
void ScalarLayerNormBaseline(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    size_t norm_size,
    float epsilon,
    bool simplified) {
  float mean = 0.0f;
  float std_dev = 0.0f;

  if (simplified) {
    float sum_sq = 0.0f;
    for (size_t h = 0; h < norm_size; h++) {
      output[h] = input[h];
      sum_sq += input[h] * input[h];
    }
    std_dev = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
  } else {
    float m2 = 0.0f;
    for (size_t h = 0; h < norm_size; h++) {
      output[h] = input[h];
      float delta = input[h] - mean;
      mean += delta / static_cast<float>(h + 1);
      float delta2 = input[h] - mean;
      m2 += delta * delta2;
    }
    std_dev = std::sqrt(m2 / static_cast<float>(norm_size) + epsilon);
  }

  for (size_t h = 0; h < norm_size; h++) {
    if (simplified) {
      output[h] = output[h] / std_dev * scale[h];
    } else if (bias == nullptr) {
      output[h] = (output[h] - mean) / std_dev * scale[h];
    } else {
      output[h] = (output[h] - mean) / std_dev * scale[h] + bias[h];
    }
  }
}

const char* DispatchPathLabel() {
#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)
  // Compile-time only: MlasLayerNormF32 always resolves to
  // MlasLayerNormKernelAppleAccelerate on this configuration, see the
  // registration in platform.cpp's MLAS_TARGET_ARM64 block.
  return "apple_accelerate_vdsp";
#else
  return GetMlasPlatform().LayerNormF32Kernel != nullptr ? "other_simd_kernel" : "no_kernel_registered";
#endif
}

// MlasLayerNormF32 returns false and does nothing (not even a scalar
// fallback -- see layernorm.cpp's MlasLayerNormF32) when no
// LayerNormF32Kernel is registered for the current platform. Without this
// check, BM_LayerNormDispatch/BM_RMSNormDispatch would silently time a
// no-op on every such platform/config and report a bogus, extremely-fast
// throughput number instead of skipping.
bool HasRegisteredKernel() {
  return GetMlasPlatform().LayerNormF32Kernel != nullptr;
}

void RunLayerNormBenchmark(
    benchmark::State& state,
    bool simplified,
    bool with_bias,
    bool use_dispatch) {
  const auto n = static_cast<size_t>(state.range(0));

  if (use_dispatch && !HasRegisteredKernel()) {
    // No LayerNormF32Kernel registered on this platform/config: skip rather
    // than silently timing MlasLayerNormF32's early "return false" no-op
    // and reporting a fake, extremely-fast throughput number for it.
    state.SkipWithMessage("no LayerNormF32Kernel registered for this platform/config");
    return;
  }

  std::vector<float> input(n), scale(n), bias(n), output(n);
  for (size_t i = 0; i < n; i++) {
    input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
    scale[i] = 1.0f + (static_cast<float>(i % 31) - 15.0f) * 0.001f;
    bias[i] = (static_cast<float>(i % 17) - 8.0f) * 0.005f;
  }
  const float* bias_ptr = (with_bias && !simplified) ? bias.data() : nullptr;

  auto run_once = [&]() -> bool {
    if (use_dispatch) {
      // The public entry point every real caller (LayerNormalization,
      // SkipLayerNormalization, SimplifiedLayerNormalization) uses.
      return MlasLayerNormF32(input.data(), scale.data(), bias_ptr, output.data(), nullptr, nullptr, n, 1e-5f,
                              simplified);
    }
    ScalarLayerNormBaseline(input.data(), scale.data(), bias_ptr, output.data(), n, 1e-5f, simplified);
    return true;
  };

  if (!run_once()) {
    state.SkipWithMessage("LayerNormF32Kernel declined this size; caller uses scalar fallback");
    return;
  }

  state.SetLabel(use_dispatch ? DispatchPathLabel() : "portable_scalar_baseline");

  for (auto _ : state) {
    bool used = run_once();
    benchmark::DoNotOptimize(used);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  const int64_t bytes_per_iteration =
      static_cast<int64_t>(n) * static_cast<int64_t>(sizeof(float)) * (bias_ptr != nullptr ? 4 : 3);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_per_iteration);
}

static void LayerNormBenchmarkArgs(benchmark::internal::Benchmark* b) {
  // Small sizes first (crossover probe), then representative real
  // transformer hidden dims. 12288 and 16384 exceed
  // kApplePerRowStackScratch (8192, see layernorm.cpp) and therefore
  // exercise the heap-fallback scratch-buffer path -- otherwise completely
  // unbenchmarked -- at a size relevant to GPT-3-class (12288) and larger
  // hidden dims.
  for (int n : {1, 8, 64, 256, 768, 1024, 1600, 2048, 3072, 4096, 8192, 12288, 16384}) {
    b->Arg(n);
  }
}

void BM_LayerNormDispatch(benchmark::State& state) {
  RunLayerNormBenchmark(state, /*simplified=*/false, /*with_bias=*/true, /*use_dispatch=*/true);
}

void BM_LayerNormPortableScalarBaseline(benchmark::State& state) {
  RunLayerNormBenchmark(state, /*simplified=*/false, /*with_bias=*/true, /*use_dispatch=*/false);
}

void BM_RMSNormDispatch(benchmark::State& state) {
  RunLayerNormBenchmark(state, /*simplified=*/true, /*with_bias=*/false, /*use_dispatch=*/true);
}

void BM_RMSNormPortableScalarBaseline(benchmark::State& state) {
  RunLayerNormBenchmark(state, /*simplified=*/true, /*with_bias=*/false, /*use_dispatch=*/false);
}

}  // namespace

BENCHMARK(BM_LayerNormDispatch)->Apply(LayerNormBenchmarkArgs)->UseRealTime();
BENCHMARK(BM_LayerNormPortableScalarBaseline)->Apply(LayerNormBenchmarkArgs)->UseRealTime();
BENCHMARK(BM_RMSNormDispatch)->Apply(LayerNormBenchmarkArgs)->UseRealTime();
BENCHMARK(BM_RMSNormPortableScalarBaseline)->Apply(LayerNormBenchmarkArgs)->UseRealTime();
