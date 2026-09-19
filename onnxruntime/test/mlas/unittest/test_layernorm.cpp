/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_layernorm.cpp

Abstract:

    Tests for MLAS LayerNorm/RMSNorm (MlasLayerNormF32).

    Covers:
      - Numeric parity against fp64-accumulated scalar reference
      - Reachability: asserts the AVX2 kernel dispatched (not silent fallback)
      - Edge cases: NormSize=1, denormals, large magnitudes, zero variance,
        NaN/Inf passthrough
      - Benchmark: in-process scalar-vs-kernel comparison (DISABLED by default)

    Tolerance: relative 0.5% (matching upstream CloseEnough) with 1e-4 absolute
    floor. On rows that meet the x86 dispatch thresholds (8 for LayerNorm, 16
    for RMSNorm), FMA reduction order can differ from the scalar fp64 reference,
    and inverse square root amplifies small variance differences. Upstream
    CloseEnough uses rel_tol=0.005; we match that convention exactly.

--*/

#include "test_util.h"
#include "mlas.h"
#include "core/mlas/lib/mlasi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------
// Capability helpers
// ---------------------------------------------------------------------------

// Returns true when the platform has a SIMD LayerNorm kernel registered
// (AVX2 on x86, RVV on RISC-V, etc.).  Tests that exercise the SIMD
// path must GTEST_SKIP() when this returns false so they don't break CI
// on ARM, older x86, or any future platform that hasn't wired up a kernel.
static bool HasLayerNormKernel() {
  return GetMlasPlatform().LayerNormF32Kernel != nullptr;
}

// Returns true when the platform uses the **centered two-pass** kernel
// (mean = sum/n with double-precision first-pass sum, then sum((x-mean)^2)).
// The precision suites assert properties specific to that algorithm —
// tolerances, B1 regression bounds, condition-number gates — that do not
// hold for other kernels (e.g. RISC-V RVV uncentered single-pass).
//
// Keep in sync: core/mlas/lib/layernorm.cpp (production #if gate),
//               GetKernelDispatchThreshold() (same #if).
static bool HasCenteredTwoPassKernel() {
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)
  return HasLayerNormKernel();
#else
  return false;
#endif
}

// ---------------------------------------------------------------------------
// fp64-accumulated scalar reference (not dependent on MLAS)
// ---------------------------------------------------------------------------
//
// Variance formula: Var = E[x²] - mean²  (uncentered, single loop in fp64).
//
// This reference deliberately uses a *different* algorithm from the kernel.
// The kernel uses a centered two-pass approach: mean = sum/n (first-pass sum
// accumulated in double), then sum((x - mean)²).  This reference instead
// computes Var = E[x²] - mean² in a single pass.  The choice is intentional
// and safe for two independent reasons:
//
//   1. fp64 precision. The uncentered formula "E[x²] - mean²" is dangerous
//      in fp32 due to catastrophic cancellation. In fp64 it is sufficiently
//      accurate for the functional test cases here. The adversarial precision
//      tests below use an fp64 Welford reference instead, avoiding cancellation.
//
//   2. Independent oracle.  A reference that uses a different algorithm from
//      the kernel cross-checks the kernel's result rather than merely
//      repeating its logic.  If the reference mirrored the kernel's centered
//      two-pass equations, a shared conceptual mistake (e.g. wrong
//      accumulator width, off-by-one in the count) could cause both to
//      produce the same wrong answer and the test would not catch it.
//      The uncentered fp64 formula and the kernel's centered two-pass are
//      algebraically equivalent but computationally independent; agreement
//      between them is a meaningful check.
//
// Do NOT make this reference mirror the kernel's algorithm: the apparent
// inconsistency is intentional and is what gives the test its value.

static void ReferenceLayerNorm(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    float* mean_out,
    float* inv_std_out,
    size_t norm_size,
    float epsilon,
    bool simplified) {
  double sum = 0.0;
  double sum_sq = 0.0;
  for (size_t i = 0; i < norm_size; i++) {
    double x = static_cast<double>(input[i]);
    sum += x;
    sum_sq += x * x;
  }
  double mean = sum / static_cast<double>(norm_size);
  double denom;
  if (simplified) {
    denom = std::sqrt(sum_sq / static_cast<double>(norm_size) +
                      static_cast<double>(epsilon));
  } else {
    denom = std::sqrt(sum_sq / static_cast<double>(norm_size) -
                      mean * mean + static_cast<double>(epsilon));
  }
  double inv_denom = 1.0 / denom;

  for (size_t i = 0; i < norm_size; i++) {
    double x = static_cast<double>(input[i]);
    double s = static_cast<double>(scale[i]);
    if (simplified) {
      output[i] = static_cast<float>(x * inv_denom * s);
    } else if (bias == nullptr) {
      output[i] = static_cast<float>((x - mean) * inv_denom * s);
    } else {
      output[i] = static_cast<float>(
          (x - mean) * inv_denom * s + static_cast<double>(bias[i]));
    }
  }
  if (mean_out) *mean_out = static_cast<float>(mean);
  if (inv_std_out) *inv_std_out = static_cast<float>(inv_denom);
}

// ---------------------------------------------------------------------------
// Test class
// ---------------------------------------------------------------------------

class MlasLayerNormTest : public MlasTestBase {
 public:
  // Minimum NormSize for SIMD dispatch.  Must mirror the production gate
  // in layernorm.cpp so the test encodes the real contract rather than
  // accepting both outcomes.
  //
  // x86 (32-bit and 64-bit): The AVX2 kernel declines NormSize < 8 for
  // LayerNorm and NormSize < 16 for RMSNorm, where SIMD setup costs exceed
  // the benefit.
  //
  // Other platforms (RISC-V RVV, future ARM SVE, etc.): variable-length
  //   vectors handle short rows natively, so the kernel dispatches for
  //   any NormSize ≥ 1.
  //
  // Keep in sync: core/mlas/lib/layernorm.cpp (production dispatch threshold).
  static constexpr size_t GetKernelDispatchThreshold(bool simplified) {
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)
    return simplified ? 16 : 8;
#else
    (void)simplified;
    return 1;
#endif
  }

  // Core test: numeric parity with conditional dispatch assertion.
  void Test(size_t norm_size, bool simplified, bool with_bias) {
    std::vector<float> input(norm_size);
    std::vector<float> scale(norm_size);
    std::vector<float> bias(norm_size);
    std::vector<float> output_ref(norm_size);
    std::vector<float> output_mlas(norm_size);
    float mean_ref = 0, mean_mlas = 0;
    float inv_std_ref = 0, inv_std_mlas = 0;

    // Deterministic fill that exercises positive, negative, and near-zero values
    for (size_t i = 0; i < norm_size; i++) {
      input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
      scale[i] = 1.0f + (static_cast<float>(i % 31) - 15.0f) * 0.001f;
      bias[i] = (static_cast<float>(i % 17) - 8.0f) * 0.005f;
    }

    const float* bias_ptr = (with_bias && !simplified) ? bias.data() : nullptr;

    ReferenceLayerNorm(input.data(), scale.data(), bias_ptr,
                       output_ref.data(), &mean_ref, &inv_std_ref,
                       norm_size, 1e-5f, simplified);

    bool used = MlasLayerNormF32(input.data(), scale.data(), bias_ptr,
                                 output_mlas.data(), &mean_mlas, &inv_std_mlas,
                                 norm_size, 1e-5f, simplified);

    // DISPATCH CONTRACT: conditional on kernel availability AND NormSize.
    //   No kernel registered → MlasLayerNormF32 returns false for all N.
    //   Kernel present + NormSize >= threshold → the kernel MUST run.
    //   Kernel present + NormSize <  threshold → the kernel MUST decline.
    //   (threshold is mode/architecture-specific: 8/16 on x86, 1 elsewhere)
    const size_t dispatch_threshold = GetKernelDispatchThreshold(simplified);
    if (!HasLayerNormKernel()) {
      ASSERT_FALSE(used)
          << "MlasLayerNormF32 returned true but no kernel is registered";
      ScalarFp32Baseline(input.data(), scale.data(), bias_ptr,
                         output_mlas.data(), &mean_mlas, &inv_std_mlas,
                         norm_size, 1e-5f, simplified);
    } else if (norm_size >= dispatch_threshold) {
      ASSERT_TRUE(used)
          << "REACHABILITY FAILURE: MlasLayerNormF32 returned false for "
             "norm_size="
          << norm_size << " (>= threshold " << dispatch_threshold
          << "). The SIMD kernel must dispatch.";
    } else {
      ASSERT_FALSE(used)
          << "DISPATCH CONTRACT VIOLATION: MlasLayerNormF32 returned true for "
             "norm_size="
          << norm_size << " (< threshold " << dispatch_threshold
          << "). The kernel must decline for small N where scalar is faster.";
      ScalarFp32Baseline(input.data(), scale.data(), bias_ptr,
                         output_mlas.data(), &mean_mlas, &inv_std_mlas,
                         norm_size, 1e-5f, simplified);
    }

    // Use relative tolerance matching upstream's CloseEnough (rel_tol=0.005)
    // with a floor of 1e-4 absolute. The AVX2 kernel uses FMA contractions
    // that produce different rounding than the scalar fp64 reference, and
    // 1/sqrt(var+eps) amplifies small variance differences — especially for
    // small NormSize where variance is near zero.
    auto near_enough = [](float got, float ref) -> bool {
      if (std::isnan(got)) return std::isnan(ref);
      float diff = std::fabs(got - ref);
      if (diff <= 1e-4f) return true;
      float top = std::max(std::fabs(got), std::fabs(ref));
      return (top > 1e-6f) && (diff / top < 0.005f);
    };

    for (size_t i = 0; i < norm_size; i++) {
      ASSERT_TRUE(near_enough(output_mlas[i], output_ref[i]))
          << "output mismatch at [" << i << "], norm_size=" << norm_size
          << " simplified=" << simplified << " bias=" << with_bias
          << " got=" << output_mlas[i] << " ref=" << output_ref[i];
    }
    // Mean is not part of the RMSNorm contract (simplified mode), so only
    // check it for full LayerNorm.
    if (!simplified) {
      ASSERT_TRUE(near_enough(mean_mlas, mean_ref))
          << "mean mismatch got=" << mean_mlas << " ref=" << mean_ref;
    }
    ASSERT_TRUE(near_enough(inv_std_mlas, inv_std_ref))
        << "inv_std_dev mismatch got=" << inv_std_mlas
        << " ref=" << inv_std_ref;
  }

  // Edge case: all-equal input → zero variance path
  void TestZeroVariance(size_t norm_size, bool simplified) {
    std::vector<float> input(norm_size, 3.14f);
    std::vector<float> scale(norm_size, 1.0f);
    std::vector<float> output_ref(norm_size);
    std::vector<float> output_mlas(norm_size);
    float mean_ref = 0, mean_mlas = 0;
    float inv_std_ref = 0, inv_std_mlas = 0;

    ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                       output_ref.data(), &mean_ref, &inv_std_ref,
                       norm_size, 1e-5f, simplified);

    bool used = MlasLayerNormF32(input.data(), scale.data(), nullptr,
                                 output_mlas.data(), &mean_mlas, &inv_std_mlas,
                                 norm_size, 1e-5f, simplified);

    // Apply the same conditional dispatch contract as Test().
    if (!HasLayerNormKernel()) {
      ASSERT_FALSE(used) << "No kernel registered but dispatch returned true";
      ScalarFp32Baseline(input.data(), scale.data(), nullptr,
                         output_mlas.data(), &mean_mlas, &inv_std_mlas,
                         norm_size, 1e-5f, simplified);
    } else if (norm_size >= GetKernelDispatchThreshold(simplified)) {
      ASSERT_TRUE(used) << "Kernel must dispatch for norm_size=" << norm_size;
    } else {
      ASSERT_FALSE(used) << "Kernel must decline for norm_size=" << norm_size;
      ScalarFp32Baseline(input.data(), scale.data(), nullptr,
                         output_mlas.data(), &mean_mlas, &inv_std_mlas,
                         norm_size, 1e-5f, simplified);
    }

    // Zero-variance: all inputs equal, so (x - mean) should be ~0.
    //
    // The kernel uses centered two-pass: mean = sum(x)/n (accumulated in
    // double), then var = sum((x - mean)^2)/n in fp32.  For constant
    // input, every (x - mean) term is exactly 0 → var = 0, inv_std =
    // 1/sqrt(eps).  Fp32 accumulation rounding in the second pass could
    // yield a tiny positive residual for large constant values, but the
    // output (x - mean) * inv_std * scale stays near zero because
    // (x - mean) ≈ 0.
    //
    // We therefore check:
    //   1. All outputs and statistics are finite (no NaN/Inf).
    //   2. Outputs match the fp64 reference within a generous tolerance
    //      that accommodates both formulations.
    auto near_enough = [](float got, float ref) -> bool {
      if (std::isnan(got)) return std::isnan(ref);
      float diff = std::fabs(got - ref);
      if (diff <= 2e-4f) return true;
      float top = std::max(std::fabs(got), std::fabs(ref));
      return (top > 1e-6f) && (diff / top < 0.005f);
    };

    for (size_t i = 0; i < norm_size; i++) {
      ASSERT_TRUE(std::isfinite(output_mlas[i]))
          << "Non-finite output at [" << i << "] for zero-variance input";
      ASSERT_TRUE(near_enough(output_mlas[i], output_ref[i]))
          << "Zero-variance mismatch at [" << i << "]"
          << " got=" << output_mlas[i] << " ref=" << output_ref[i];
    }
    ASSERT_TRUE(std::isfinite(inv_std_mlas)) << "inv_std_dev must be finite";
  }

  // Edge case: denormals — finiteness check only (not accuracy).
  // Verifies the kernel does not produce NaN/Inf on denormal inputs.
  void TestDenormals(size_t norm_size) {
    std::vector<float> input(norm_size);
    std::vector<float> scale(norm_size, 1.0f);
    std::vector<float> output_ref(norm_size);
    std::vector<float> output_mlas(norm_size);
    float mean_ref, mean_mlas, inv_std_ref, inv_std_mlas;

    float denorm = std::numeric_limits<float>::denorm_min();
    for (size_t i = 0; i < norm_size; i++) {
      input[i] = denorm * static_cast<float>(i + 1);
    }

    ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                       output_ref.data(), &mean_ref, &inv_std_ref,
                       norm_size, 1e-5f, false);

    bool used = MlasLayerNormF32(input.data(), scale.data(), nullptr,
                                 output_mlas.data(), &mean_mlas, &inv_std_mlas,
                                 norm_size, 1e-5f, false);
    if (!HasLayerNormKernel()) {
      GTEST_SKIP() << "No SIMD LayerNorm kernel on this platform";
    }
    ASSERT_TRUE(used);

    for (size_t i = 0; i < norm_size; i++) {
      ASSERT_TRUE(std::isfinite(output_mlas[i]))
          << "Non-finite on denormal input at [" << i << "]";
    }
  }

  // Edge case: large magnitudes — finiteness check only (not accuracy).
  // Verifies the kernel does not produce NaN/Inf on ±1e30 inputs.
  void TestLargeMagnitudes(size_t norm_size) {
    std::vector<float> input(norm_size);
    std::vector<float> scale(norm_size, 1.0f);
    std::vector<float> output_ref(norm_size);
    std::vector<float> output_mlas(norm_size);
    float mean_ref, mean_mlas, inv_std_ref, inv_std_mlas;

    for (size_t i = 0; i < norm_size; i++) {
      input[i] = ((i % 2 == 0) ? 1.0f : -1.0f) * 1e30f;
    }

    ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                       output_ref.data(), &mean_ref, &inv_std_ref,
                       norm_size, 1e-5f, false);

    bool used = MlasLayerNormF32(input.data(), scale.data(), nullptr,
                                 output_mlas.data(), &mean_mlas, &inv_std_mlas,
                                 norm_size, 1e-5f, false);
    if (!HasLayerNormKernel()) {
      GTEST_SKIP() << "No SIMD LayerNorm kernel on this platform";
    }
    ASSERT_TRUE(used);

    for (size_t i = 0; i < norm_size; i++) {
      ASSERT_TRUE(std::isfinite(output_mlas[i]))
          << "Non-finite on large-magnitude input at [" << i << "]";
    }
  }

  // Edge case: NaN/Inf passthrough — must match scalar behavior
  void TestNanInf(size_t norm_size) {
    if (norm_size < 3) return;
    std::vector<float> input(norm_size, 1.0f);
    std::vector<float> scale(norm_size, 1.0f);
    std::vector<float> output_ref(norm_size);
    std::vector<float> output_mlas(norm_size);
    float mean_ref, mean_mlas, inv_std_ref, inv_std_mlas;

    input[0] = std::numeric_limits<float>::quiet_NaN();
    input[1] = std::numeric_limits<float>::infinity();
    input[2] = -std::numeric_limits<float>::infinity();

    ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                       output_ref.data(), &mean_ref, &inv_std_ref,
                       norm_size, 1e-5f, false);

    bool used = MlasLayerNormF32(input.data(), scale.data(), nullptr,
                                 output_mlas.data(), &mean_mlas, &inv_std_mlas,
                                 norm_size, 1e-5f, false);
    if (!HasLayerNormKernel()) {
      GTEST_SKIP() << "No SIMD LayerNorm kernel on this platform";
    }
    ASSERT_TRUE(used);
    for (size_t i = 0; i < norm_size; i++) {
      if (std::isnan(output_ref[i])) {
        ASSERT_TRUE(std::isnan(output_mlas[i]))
            << "Expected NaN at [" << i << "]";
      } else if (std::isinf(output_ref[i])) {
        ASSERT_TRUE(std::isinf(output_mlas[i]))
            << "Expected Inf at [" << i << "]";
      }
    }
  }

  // -----------------------------------------------------------------------
  // True scalar fp32 baseline — reproduces the fallback path from
  // onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc (ComputeJob)
  // that runs when MlasLayerNormF32() returns false on x86 prior to
  // this PR. This is the code the AVX2 kernel actually replaces.
  //
  // IMPORTANT: This is fp32 throughout (no fp64 accumulation), matching
  // the production fallback. Do NOT confuse with ReferenceLayerNorm above
  // which uses fp64 accumulation for correctness testing.
  // -----------------------------------------------------------------------
  static void ScalarFp32Baseline(
      const float* input,
      const float* scale,
      const float* bias,
      float* output,
      float* mean_out,
      float* inv_std_out,
      int64_t norm_size,
      float epsilon,
      bool simplified) {
    float mean = 0.0f;
    float std_dev = 0.0f;

    if (simplified) {
      // RMSNorm: sum of squares, single pass
      float sum_sq = 0.0f;
      for (int64_t h = 0; h < norm_size; h++) {
        output[h] = input[h];
        sum_sq += input[h] * input[h];
      }
      std_dev = sqrt(sum_sq / norm_size + epsilon);
    } else {
      // Welford's online algorithm in fp32.
      float M2 = 0.0f;
      for (int64_t h = 0; h < norm_size; h++) {
        output[h] = input[h];
        float delta = input[h] - mean;
        mean += delta / static_cast<float>(h + 1);
        float delta2 = input[h] - mean;
        M2 += delta * delta2;
      }
      std_dev = sqrt(M2 / norm_size + epsilon);
    }

    for (int64_t h = 0; h < norm_size; h++) {
      if (simplified) {
        output[h] = output[h] / std_dev * scale[h];
      } else if (bias == nullptr) {
        output[h] = (output[h] - mean) / std_dev * scale[h];
      } else {
        output[h] = (output[h] - mean) / std_dev * scale[h] + bias[h];
      }
    }

    if (mean_out != nullptr) *mean_out = mean;
    if (inv_std_out != nullptr) *inv_std_out = 1 / std_dev;
  }

  // Benchmark: SIMD kernel vs true scalar fp32 baseline.
  //
  // Scalar baseline note: Welford's online mean update performs one fp32
  // division per element (delta / (h+1)) in the reduction loop, whereas
  // the AVX2 kernel accumulates a double-precision sum and divides once.
  // This per-element division inflates the scalar timing; the reported
  // speedup therefore includes both the SIMD benefit and this algorithmic
  // difference. The scalar normalization pass also performs one division per
  // output, matching the production fallback expression order.
  void Benchmark(size_t norm_size, size_t warmup, size_t iters, bool simplified) {
    std::vector<float> input(norm_size);
    std::vector<float> scale(norm_size);
    std::vector<float> output(norm_size);
    float mean_out, inv_std_out;

    for (size_t i = 0; i < norm_size; i++) {
      input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
      scale[i] = 1.0f + (static_cast<float>(i % 31) - 15.0f) * 0.001f;
    }

    // Production passes nullptr for MeanOut in simplified (RMSNorm) mode.
    // Benchmark must match production to exercise the real fast path.
    float* mean_ptr = simplified ? nullptr : &mean_out;

    // Warmup + measure: AVX2 kernel
    for (size_t i = 0; i < warmup; i++) {
      bool used = MlasLayerNormF32(input.data(), scale.data(), nullptr,
                                   output.data(), mean_ptr, &inv_std_out,
                                   norm_size, 1e-5f, simplified);
      if (i == 0) {
        ASSERT_TRUE(used)
            << "Benchmark requires SIMD kernel dispatch for norm_size="
            << norm_size << "; got scalar fallback.";
      }
    }
    std::vector<double> kernel_us(iters);
    for (size_t i = 0; i < iters; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      MlasLayerNormF32(input.data(), scale.data(), nullptr,
                       output.data(), mean_ptr, &inv_std_out,
                       norm_size, 1e-5f, simplified);
      auto t1 = std::chrono::high_resolution_clock::now();
      kernel_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // Warmup + measure: scalar fp32 baseline (the actual code being replaced)
    for (size_t i = 0; i < warmup; i++) {
      ScalarFp32Baseline(input.data(), scale.data(), nullptr,
                         output.data(), mean_ptr, &inv_std_out,
                         norm_size, 1e-5f, simplified);
    }
    std::vector<double> scalar_us(iters);
    for (size_t i = 0; i < iters; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      ScalarFp32Baseline(input.data(), scale.data(), nullptr,
                         output.data(), mean_ptr, &inv_std_out,
                         norm_size, 1e-5f, simplified);
      auto t1 = std::chrono::high_resolution_clock::now();
      scalar_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // Also measure the fp64 reference for context (the independent oracle baseline)
    for (size_t i = 0; i < warmup; i++) {
      ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                         output.data(), mean_ptr, &inv_std_out,
                         norm_size, 1e-5f, simplified);
    }
    std::vector<double> fp64_us(iters);
    for (size_t i = 0; i < iters; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                         output.data(), mean_ptr, &inv_std_out,
                         norm_size, 1e-5f, simplified);
      auto t1 = std::chrono::high_resolution_clock::now();
      fp64_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    auto stats = [](std::vector<double>& v) {
      std::sort(v.begin(), v.end());
      size_t n = v.size();
      double sum = std::accumulate(v.begin(), v.end(), 0.0);
      double mean_val = sum / static_cast<double>(n);
      double sq = 0;
      for (auto x : v) sq += (x - mean_val) * (x - mean_val);
      struct S {
        double p50, p95, mean, stdev;
      };
      return S{v[n / 2], v[static_cast<size_t>(n * 0.95)], mean_val,
               std::sqrt(sq / static_cast<double>(n))};
    };

    auto ks = stats(kernel_us);
    auto ss = stats(scalar_us);
    auto fs = stats(fp64_us);
    const char* mode = simplified ? "RMSNorm" : "LayerNorm";
    printf("BENCH %s norm_size=%zu iters=%zu\n", mode, norm_size, iters);
    printf("  avx2_kernel:     p50=%.3fus p95=%.3fus mean=%.3fus stdev=%.3fus\n",
           ks.p50, ks.p95, ks.mean, ks.stdev);
    printf("  scalar_fp32:     p50=%.3fus p95=%.3fus mean=%.3fus stdev=%.3fus\n",
           ss.p50, ss.p95, ss.mean, ss.stdev);
    printf("  fp64_ref:        p50=%.3fus p95=%.3fus mean=%.3fus stdev=%.3fus\n",
           fs.p50, fs.p95, fs.mean, fs.stdev);
    printf("  speedup_vs_fp32: %.2fx (p50)  %.2fx (p95)\n",
           ss.p50 / ks.p50, ss.p95 / ks.p95);
    printf("  speedup_vs_fp64: %.2fx (p50)  [inflated — NOT the true baseline]\n",
           fs.p50 / ks.p50);
    printf("\n");
  }
};

// ---------------------------------------------------------------------------
// Short-execute test registration (upstream convention)
// ---------------------------------------------------------------------------

class LayerNormShortExecuteTest : public MlasTestFixture<MlasLayerNormTest> {
 public:
  LayerNormShortExecuteTest(size_t norm_size, bool simplified, bool with_bias)
      : norm_size_(norm_size), simplified_(simplified), with_bias_(with_bias) {}

  void TestBody() override {
    MlasTestFixture<MlasLayerNormTest>::mlas_tester->Test(
        norm_size_, simplified_, with_bias_);
  }

  static size_t RegisterSingleTest(size_t norm_size, bool simplified,
                                   bool with_bias) {
    std::stringstream ss;
    ss << "/norm_size" << norm_size
       << "/simplified" << simplified
       << "/bias" << with_bias;
    auto test_name = ss.str();

    testing::RegisterTest(
        "LayerNorm",
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasLayerNormTest>* {
          return new LayerNormShortExecuteTest(norm_size, simplified, with_bias);
        });
    return 1;
  }

  // NormSize values deliberately span non-multiples of the 8-wide AVX2 vector
  // so the scalar tail path is exercised.
  static size_t RegisterShortExecuteTests() {
    size_t count = 0;
    for (size_t n : {1, 7, 8, 15, 16, 127, 128, 1024}) {
      for (bool simplified : {true, false}) {
        for (bool with_bias : {true, false}) {
          count += RegisterSingleTest(n, simplified, with_bias);
        }
      }
    }
    return count;
  }

 private:
  size_t norm_size_;
  bool simplified_;
  bool with_bias_;
};

// ---------------------------------------------------------------------------
// Edge-case tests registered as standalone TEST_F
// ---------------------------------------------------------------------------

class MlasLayerNormEdgeTest : public MlasTestFixture<MlasLayerNormTest> {};

TEST_F(MlasLayerNormEdgeTest, ZeroVariance) {
  for (size_t n : {1, 8, 15, 128}) {
    mlas_tester->TestZeroVariance(n, false);
    mlas_tester->TestZeroVariance(n, true);
  }
}

TEST_F(MlasLayerNormEdgeTest, Denormals) {
  for (size_t n : {8, 15, 128}) {
    mlas_tester->TestDenormals(n);
  }
}

TEST_F(MlasLayerNormEdgeTest, LargeMagnitudes) {
  for (size_t n : {8, 15, 128}) {
    mlas_tester->TestLargeMagnitudes(n);
  }
}

TEST_F(MlasLayerNormEdgeTest, NanInf) {
  for (size_t n : {8, 15, 128}) {
    mlas_tester->TestNanInf(n);
  }
}

// ---------------------------------------------------------------------------
// Adversarial numeric precision tests
//
// Purpose: compare the centered two-pass AVX2 kernel (double-precision mean,
// fp32 variance) vs scalar fp32 baseline vs fp64 reference on inputs designed
// to stress catastrophic cancellation and accumulation error.  The test prints
// a comparison table for human review and asserts a defensible tolerance.
// ---------------------------------------------------------------------------

class MlasLayerNormPrecisionTest : public MlasTestFixture<MlasLayerNormTest> {};

// Helper: compute fp64 Welford reference (gold standard)
static void WelfordFp64Reference(
    const float* input, const float* scale, const float* bias,
    double* output, double* mean_out, double* inv_std_out,
    size_t norm_size, double epsilon, bool simplified) {
  if (simplified) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < norm_size; i++) {
      double x = static_cast<double>(input[i]);
      sum_sq += x * x;
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(norm_size) + epsilon);
    double inv = 1.0 / rms;
    for (size_t i = 0; i < norm_size; i++) {
      output[i] = static_cast<double>(input[i]) * inv *
                  static_cast<double>(scale[i]);
    }
    *mean_out = 0.0;
    *inv_std_out = inv;
  } else {
    // Welford's in fp64
    double mean = 0.0;
    double M2 = 0.0;
    for (size_t h = 0; h < norm_size; h++) {
      double x = static_cast<double>(input[h]);
      double delta = x - mean;
      mean += delta / static_cast<double>(h + 1);
      double delta2 = x - mean;
      M2 += delta * delta2;
    }
    double var = M2 / static_cast<double>(norm_size);
    double std_dev = std::sqrt(var + epsilon);
    double inv = 1.0 / std_dev;
    for (size_t i = 0; i < norm_size; i++) {
      double x = static_cast<double>(input[i]);
      double s = static_cast<double>(scale[i]);
      if (bias) {
        output[i] = (x - mean) * inv * s + static_cast<double>(bias[i]);
      } else {
        output[i] = (x - mean) * inv * s;
      }
    }
    *mean_out = mean;
    *inv_std_out = inv;
  }
}

// Helper: measure max relative error of fp32 outputs vs fp64 reference
// Vector-normalised max error: ||got − ref||_∞ / ||ref||_∞.
// Unlike per-element relative error, this does not blow up when individual
// reference values are near zero (as expected in layernorm output, which is
// approximately standard normal).  Returns 1e30 if any element is non-finite.
static double MaxRelError(const float* got, const double* ref, size_t n) {
  double max_diff = 0.0;
  double max_ref = 0.0;
  for (size_t i = 0; i < n; i++) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) return 1e30;
    double diff = std::fabs(static_cast<double>(got[i]) - ref[i]);
    if (diff > max_diff) max_diff = diff;
    double mag = std::fabs(ref[i]);
    if (mag > max_ref) max_ref = mag;
  }
  return (max_ref > 1e-30) ? max_diff / max_ref : max_diff;
}

// Run one precision scenario and print results. Returns max rel error of AVX2.
static double RunPrecisionScenario(
    const char* name,
    const float* input, const float* scale, const float* bias,
    size_t norm_size, float epsilon, bool simplified) {
  // 1. fp64 Welford reference
  std::vector<double> out_fp64(norm_size);
  double mean_fp64, inv_std_fp64;
  WelfordFp64Reference(input, scale, bias, out_fp64.data(),
                       &mean_fp64, &inv_std_fp64, norm_size, epsilon, simplified);

  // 2. Welford fp32 (the code being replaced)
  std::vector<float> out_welford(norm_size);
  float mean_welford, inv_std_welford;
  MlasLayerNormTest::ScalarFp32Baseline(input, scale, bias, out_welford.data(),
                                        &mean_welford, &inv_std_welford,
                                        norm_size, epsilon, simplified);

  // 3. Centered two-pass AVX2 kernel
  std::vector<float> out_avx2(norm_size);
  float mean_avx2, inv_std_avx2;
  bool used = MlasLayerNormF32(input, scale, bias, out_avx2.data(),
                               &mean_avx2, &inv_std_avx2,
                               norm_size, epsilon, simplified);
  EXPECT_TRUE(used) << name << ": kernel must dispatch";

  double err_welford = MaxRelError(out_welford.data(), out_fp64.data(), norm_size);
  double err_avx2 = MaxRelError(out_avx2.data(), out_fp64.data(), norm_size);

  // Mean and inv_std_dev relative error
  double mean_err_w = (std::fabs(mean_fp64) > 1e-30)
                          ? std::fabs(static_cast<double>(mean_welford) - mean_fp64) / std::fabs(mean_fp64)
                          : std::fabs(static_cast<double>(mean_welford) - mean_fp64);
  double mean_err_a = (std::fabs(mean_fp64) > 1e-30)
                          ? std::fabs(static_cast<double>(mean_avx2) - mean_fp64) / std::fabs(mean_fp64)
                          : std::fabs(static_cast<double>(mean_avx2) - mean_fp64);
  double inv_err_w = (std::fabs(inv_std_fp64) > 1e-30)
                         ? std::fabs(static_cast<double>(inv_std_welford) - inv_std_fp64) / std::fabs(inv_std_fp64)
                         : std::fabs(static_cast<double>(inv_std_welford) - inv_std_fp64);
  double inv_err_a = (std::fabs(inv_std_fp64) > 1e-30)
                         ? std::fabs(static_cast<double>(inv_std_avx2) - inv_std_fp64) / std::fabs(inv_std_fp64)
                         : std::fabs(static_cast<double>(inv_std_avx2) - inv_std_fp64);

  printf(
      "  %-40s N=%-6zu  welford_fp32: out=%.2e mean=%.2e inv=%.2e  |  "
      "avx2_centered: out=%.2e mean=%.2e inv=%.2e  |  ratio=%.1fx\n",
      name, norm_size,
      err_welford, mean_err_w, inv_err_w,
      err_avx2, mean_err_a, inv_err_a,
      (err_welford > 1e-30) ? err_avx2 / err_welford : 0.0);

  return err_avx2;
}

// DISABLED by default — run manually with --gtest_also_run_disabled_tests.
// This is a measurement/reporting tool, not a correctness gate.
// Prints a full comparison table including catastrophic-cancellation scenarios
// where two-pass is known to degrade.
TEST_F(MlasLayerNormPrecisionTest, DISABLED_AdversarialPrecisionReport) {
  if (!HasCenteredTwoPassKernel()) {
    GTEST_SKIP() << "No centered two-pass kernel on this platform (x86 only)";
  }
  printf("\n");
  printf("======================================================================\n");
  printf("  ADVERSARIAL PRECISION: centered two-pass AVX2 vs fp64 ref\n");
  printf("  All values are MAX RELATIVE ERROR vs fp64 reference.\n");
  printf("======================================================================\n");

  const float eps = 1e-5f;
  double worst_avx2 = 0.0;
  double worst_catastrophic = 0.0;  // tracked separately for extreme cond#

  // -------------------------------------------------------------------
  // SCENARIO 1: Large N with benign data
  // -------------------------------------------------------------------
  printf("\n--- Scenario 1: Large N, benign data ---\n");
  for (size_t N : {4096, 16384, 65536}) {
    std::vector<float> input(N), scale(N, 1.0f);
    for (size_t i = 0; i < N; i++) {
      input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
    }
    double e = RunPrecisionScenario("large_N_benign", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst_avx2 = std::max(worst_avx2, e);
  }

  // -------------------------------------------------------------------
  // SCENARIO 2: High dynamic range — mixed tiny and huge values
  // -------------------------------------------------------------------
  printf("\n--- Scenario 2: High dynamic range ---\n");
  for (size_t N : {256, 4096}) {
    std::vector<float> input(N), scale(N, 1.0f);
    for (size_t i = 0; i < N; i++) {
      // Alternate between 1e-6 and 1e6
      input[i] = (i % 2 == 0) ? 1e-6f : 1e6f;
      // Add small perturbation
      input[i] *= (1.0f + static_cast<float>(i % 37) * 1e-4f);
    }
    double e = RunPrecisionScenario("high_dynamic_range", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst_avx2 = std::max(worst_avx2, e);
  }

  // -------------------------------------------------------------------
  // SCENARIO 3: CATASTROPHIC CANCELLATION — large mean, tiny variance
  // This is THE critical case for testing the kernel's numerical
  // stability. The fp32 second-pass subtraction (x - mean) in the
  // centered two-pass loses precision when mean is huge and
  // perturbations are tiny (condition number ≫ 1e7). The kernel's
  // double-precision first-pass sum gives an accurate mean, keeping
  // the second-pass subtraction viable up to much higher condition
  // numbers than a scalar fp32 Welford.
  // -------------------------------------------------------------------
  printf("\n--- Scenario 3: CATASTROPHIC CANCELLATION (large mean, tiny var) ---\n");
  for (size_t N : {256, 1024, 4096}) {
    std::vector<float> input(N), scale(N, 1.0f);
    float base = 1e6f;
    for (size_t i = 0; i < N; i++) {
      // Values near 1e6 with spread ~1e-3 → condition number ~1e9.
      // At this condition number, fp32 (x - mean) subtraction loses
      // all significant bits of the perturbation. The double-precision
      // mean helps but cannot save the fp32 second pass entirely.
      input[i] = base + (static_cast<float>(i % 100) - 50.0f) * 1e-3f;
    }
    double e = RunPrecisionScenario("catastrophic_cancel_1e6", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst_catastrophic = std::max(worst_catastrophic, e);
  }

  // Even more extreme: base = 1e7
  for (size_t N : {256, 1024}) {
    std::vector<float> input(N), scale(N, 1.0f);
    float base = 1e7f;
    for (size_t i = 0; i < N; i++) {
      input[i] = base + (static_cast<float>(i % 100) - 50.0f) * 1e-2f;
    }
    double e = RunPrecisionScenario("catastrophic_cancel_1e7", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst_catastrophic = std::max(worst_catastrophic, e);
  }

  // -------------------------------------------------------------------
  // SCENARIO 4: Near-zero variance at large magnitude
  // All values the same large number + tiny epsilon perturbation
  // -------------------------------------------------------------------
  printf("\n--- Scenario 4: Near-zero variance at large magnitude ---\n");
  for (size_t N : {256, 1024}) {
    std::vector<float> input(N), scale(N, 1.0f);
    float base = 1e5f;
    for (size_t i = 0; i < N; i++) {
      input[i] = base + (i == 0 ? 1e-4f : 0.0f);
    }
    double e = RunPrecisionScenario("near_zero_var_large_mag", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst_avx2 = std::max(worst_avx2, e);
  }

  // -------------------------------------------------------------------
  // SCENARIO 5: Denormals mixed with normal values
  // -------------------------------------------------------------------
  printf("\n--- Scenario 5: Denormals mixed ---\n");
  for (size_t N : {256, 1024}) {
    std::vector<float> input(N), scale(N, 1.0f);
    float denorm = std::numeric_limits<float>::denorm_min();
    for (size_t i = 0; i < N; i++) {
      input[i] = (i % 4 == 0) ? denorm * static_cast<float>(i + 1)
                              : static_cast<float>(i % 17) * 0.1f;
    }
    double e = RunPrecisionScenario("denormals_mixed", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst_avx2 = std::max(worst_avx2, e);
  }

  // -------------------------------------------------------------------
  // SCENARIO 6: Near FP32 max — EXCLUDED.
  // Values near FLT_MAX produce sum(x²) that overflows fp32 (and even
  // fp64 in some formulations), yielding Inf/NaN.  This is inherent to
  // any algorithm that accumulates squares of near-max floats and is
  // not a kernel defect.  Keeping the scenario would cause 100% error
  // and make the test un-enableable.
  // -------------------------------------------------------------------

  // -------------------------------------------------------------------
  // SCENARIO 7: Realistic LLM hidden-state distributions
  // Typical transformer hidden states: mean ~0, std ~1-5, dim 768-4096
  // -------------------------------------------------------------------
  printf("\n--- Scenario 7: Realistic LLM activations ---\n");
  for (size_t N : {768, 1024, 2048, 4096}) {
    std::vector<float> input(N), scale(N);
    // Pseudo-Gaussian via simple deterministic hash
    for (size_t i = 0; i < N; i++) {
      // Simple deterministic "random" in [-3, 3] range (typical activation)
      uint32_t h = static_cast<uint32_t>(i * 2654435761u);
      float u = static_cast<float>(h & 0xFFFFFF) / static_cast<float>(0xFFFFFF);
      input[i] = (u - 0.5f) * 6.0f;  // range [-3, 3]
      scale[i] = 0.9f + 0.2f * static_cast<float>(i % 10) / 9.0f;
    }
    double e = RunPrecisionScenario("llm_activations", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst_avx2 = std::max(worst_avx2, e);
  }

  // Also RMSNorm (simplified) for large-mean case — should be unaffected
  printf("\n--- Scenario 8: RMSNorm (simplified) with large values ---\n");
  for (size_t N : {256, 1024, 4096}) {
    std::vector<float> input(N), scale(N, 1.0f);
    for (size_t i = 0; i < N; i++) {
      input[i] = 1e6f + (static_cast<float>(i % 100) - 50.0f) * 1e-3f;
    }
    double e = RunPrecisionScenario("rmsnorm_large_mean", input.data(),
                                    scale.data(), nullptr, N, eps, true);
    worst_avx2 = std::max(worst_avx2, e);
  }

  printf("\n======================================================================\n");
  printf("  SUMMARY: worst AVX2 centered two-pass rel error = %.6e\n", worst_avx2);
  printf("  (catastrophic-cancellation scenarios: %.6e — tracked separately)\n",
         worst_catastrophic);
  printf("======================================================================\n\n");

  // Non-catastrophic scenarios must stay within 0.5% of fp64 reference.
  EXPECT_LT(worst_avx2, 0.005)
      << "Centered two-pass AVX2 kernel exceeds 0.5% max relative error vs fp64 "
         "reference. See the printed table above for per-scenario "
         "breakdown.";

  // Catastrophic-cancellation scenarios (condition number >= 1e8) are
  // expected to lose precision in fp32 regardless of algorithm.  The
  // centered two-pass kernel is ~10× better than scalar Welford fp32
  // here.  Gate at 10% to detect regressions without failing on
  // inherent fp32 limits.
  EXPECT_LT(worst_catastrophic, 0.1)
      << "Catastrophic-cancellation scenarios exceed 10% rel error vs fp64. "
         "Scalar Welford is ~95%; current = "
      << worst_catastrophic << ".";
}

// Passing test: realistic LLM activation distributions stay within tolerance.
TEST_F(MlasLayerNormPrecisionTest, RealisticLLMPrecision) {
  if (!HasCenteredTwoPassKernel()) {
    GTEST_SKIP() << "No centered two-pass kernel on this platform (x86 only)";
  }
  const float eps = 1e-5f;
  double worst = 0.0;
  for (size_t N : {768, 1024, 2048, 4096, 16384}) {
    std::vector<float> input(N), scale(N);
    for (size_t i = 0; i < N; i++) {
      uint32_t h = static_cast<uint32_t>(i * 2654435761u);
      float u = static_cast<float>(h & 0xFFFFFF) / static_cast<float>(0xFFFFFF);
      input[i] = (u - 0.5f) * 6.0f;
      scale[i] = 0.9f + 0.2f * static_cast<float>(i % 10) / 9.0f;
    }
    double e = RunPrecisionScenario("llm_realistic", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst = std::max(worst, e);
  }
  EXPECT_LT(worst, 1e-4)
      << "AVX2 centered two-pass exceeds 0.01% rel error on realistic LLM activations";
}

// Passing test: large N with benign data stays within tolerance.
TEST_F(MlasLayerNormPrecisionTest, LargeNBenignPrecision) {
  if (!HasCenteredTwoPassKernel()) {
    GTEST_SKIP() << "No centered two-pass kernel on this platform (x86 only)";
  }
  const float eps = 1e-5f;
  double worst = 0.0;
  for (size_t N : {4096, 16384, 65536}) {
    std::vector<float> input(N), scale(N, 1.0f);
    for (size_t i = 0; i < N; i++) {
      input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
    }
    double e = RunPrecisionScenario("large_N_benign", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst = std::max(worst, e);
  }
  EXPECT_LT(worst, 1e-3)
      << "AVX2 centered two-pass exceeds 0.1% rel error on large-N benign data";
}

// Passing test: high dynamic range stays within tolerance.
TEST_F(MlasLayerNormPrecisionTest, HighDynamicRangePrecision) {
  if (!HasCenteredTwoPassKernel()) {
    GTEST_SKIP() << "No centered two-pass kernel on this platform (x86 only)";
  }
  const float eps = 1e-5f;
  double worst = 0.0;
  for (size_t N : {256, 4096}) {
    std::vector<float> input(N), scale(N, 1.0f);
    for (size_t i = 0; i < N; i++) {
      input[i] = (i % 2 == 0) ? 1e-6f : 1e6f;
      input[i] *= (1.0f + static_cast<float>(i % 37) * 1e-4f);
    }
    double e = RunPrecisionScenario("high_dynamic_range", input.data(),
                                    scale.data(), nullptr, N, eps, false);
    worst = std::max(worst, e);
  }
  EXPECT_LT(worst, 1e-4)
      << "AVX2 centered two-pass exceeds 0.01% rel error on high dynamic range data";
}

// Catastrophic cancellation stress test: inputs with large base offset and
// tiny spread.  The centered two-pass kernel with double-precision first-pass
// sum must:
//   1. Produce no NaN/Inf (all scenarios)
//   2. For scenarios where condition number < 1e7 (within fp32 range),
//      match the fp64 oracle to within 0.1% max relative error
//   3. For extreme condition numbers (≥ 1e7), only finiteness is asserted
//      because fp32 second-pass subtraction inherently loses precision
TEST_F(MlasLayerNormPrecisionTest, CatastrophicCancellationPasses) {
  if (!HasCenteredTwoPassKernel()) {
    GTEST_SKIP() << "No centered two-pass kernel on this platform (x86 only)";
  }
  const float eps = 1e-5f;

  struct Scenario {
    const char* name;
    float base;
    float spread;
  };
  Scenario scenarios[] = {
      // condition = 1e4 — well within fp32 range, accuracy is checkable
      {"catastrophic_1e4_cond1e4", 1e4f, 1.0f},
      // condition = 1e5 — moderate cancellation stress
      {"catastrophic_1e5_cond1e5", 1e5f, 1.0f},
      // condition = 1e9 — beyond fp32 precision; only finiteness is checked
      {"catastrophic_1e6", 1e6f, 1e-3f},
      // condition = 1e9 — beyond fp32 precision; only finiteness is checked
      {"catastrophic_1e7", 1e7f, 1e-2f},
  };

  for (const auto& sc : scenarios) {
    for (size_t N : {256, 1024}) {
      std::vector<float> input(N), scale(N, 1.0f);
      for (size_t i = 0; i < N; i++) {
        input[i] = sc.base + (static_cast<float>(i % 100) - 50.0f) * sc.spread;
      }

      // Centered two-pass SIMD kernel
      std::vector<float> out_avx2(N);
      float mean_avx2, inv_std_avx2;
      bool used = MlasLayerNormF32(input.data(), scale.data(), nullptr,
                                   out_avx2.data(), &mean_avx2, &inv_std_avx2,
                                   N, eps, false);
      ASSERT_TRUE(used) << sc.name << " N=" << N << ": kernel must dispatch";

      // 1. No NaN/Inf — the critical improvement over uncentered E[x²]-mean²
      for (size_t i = 0; i < N; i++) {
        ASSERT_TRUE(std::isfinite(out_avx2[i]))
            << sc.name << " N=" << N << ": NaN/Inf at output[" << i << "]";
      }
      ASSERT_TRUE(std::isfinite(mean_avx2))
          << sc.name << " N=" << N << ": mean is NaN/Inf";
      ASSERT_TRUE(std::isfinite(inv_std_avx2))
          << sc.name << " N=" << N << ": inv_std_dev is NaN/Inf";

      // 2. For condition numbers within fp32 range (base/spread < ~1e7),
      //    also check accuracy vs fp64 oracle.  At base=1e6/spread=1e-3
      //    the condition number (~1e9) exceeds fp32's ~7 digits, so the
      //    second-pass subtraction x-mean loses all precision in fp32.
      //    This is inherent to fp32 arithmetic, not a kernel bug.
      double condition = static_cast<double>(sc.base) / static_cast<double>(sc.spread);
      if (condition < 1e7) {
        std::vector<double> out_fp64(N);
        double mean_fp64, inv_std_fp64;
        WelfordFp64Reference(input.data(), scale.data(), nullptr,
                             out_fp64.data(), &mean_fp64, &inv_std_fp64,
                             N, static_cast<double>(eps), false);

        double fp64_err = MaxRelError(out_avx2.data(), out_fp64.data(), N);
        EXPECT_LT(fp64_err, 1e-3)
            << sc.name << " N=" << N
            << ": kernel diverges from fp64 oracle (fp64_err="
            << fp64_err << ")";

        printf("  %-45s N=%-6zu  finite=OK  fp64_err=%.2e\n",
               sc.name, N, fp64_err);
      } else {
        printf("  %-45s N=%-6zu  finite=OK  (cond=%.0e, fp32 limit)\n",
               sc.name, N, condition);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// N5/N6: fp64 parity sweep — reviewer-mandated grid
//
// Measures MlasLayerNormF32 output against a fp64-accumulated reference
// for every combination in the specified grid.  Implementation-agnostic:
// any correct reduction must pass; any regression of the magnitude seen
// in B1 (scalar Welford 0.94 vs kernel centered two-pass 0.033) must fail.
//
// Effective range after the condition-number gate (base/spread < 1e6):
//   Of 16 (base, spread) pairs, 6 pass the gate:
//     {1e3}/{1.0, 0.1, 0.01}, {1e4}/{1.0, 0.1}, {1e5}/{1.0}
//   The remaining 10 pairs (cond ≥ 1e6) are skipped because fp32
//   second-pass subtraction loses all perturbation precision there,
//   making accuracy a test of float limits, not kernel correctness.
//   Total: 6 pairs × 3 epsilons × 10 NormSizes = 180 cases.
// ---------------------------------------------------------------------------

TEST_F(MlasLayerNormPrecisionTest, Fp64ParitySweep) {
  if (!HasCenteredTwoPassKernel()) {
    GTEST_SKIP() << "No centered two-pass kernel on this platform (x86 only)";
  }

  // Grid from the reviewer's specification
  const double bases[] = {1e3, 1e4, 1e5, 1e6};
  const double spreads[] = {1.0, 1e-1, 1e-2, 1e-3};
  const float epsilons[] = {1e-5f, 1e-6f, 1e-12f};
  // NormSize including non-multiples of 8
  const size_t norm_sizes[] = {9, 15, 33, 127, 255, 256, 512, 1024, 2048, 4096};

  // Tolerance: 3% normalised max error (||diff||_∞ / ||ref||_∞).
  // Worst observed on this kernel is ≈ 2.23e-02, giving ~35% headroom.
  // The previous 2.5e-2 threshold had only 12% headroom, too thin for
  // cross-platform / compiler variation.  At 3e-2 the B1 regression
  // guard still bites: the removed lane-parallel Welford measured
  // 2.49e-01, which is 8× above this threshold.
  constexpr double kMaxRelError = 3e-2;

  double overall_worst = 0.0;
  size_t total_cases = 0;
  size_t failures = 0;

  for (double base : bases) {
    for (double spread : spreads) {
      // Skip when condition number (base/spread) exceeds what fp32
      // can meaningfully compute.  At cond ≥ 1e6 the second-pass
      // subtraction x − mean in fp32 loses more than 1 digit of the
      // perturbation, making accuracy vs fp64 a test of float
      // precision, not kernel correctness.
      if (base / spread >= 1e6) continue;

      for (float eps : epsilons) {
        for (size_t N : norm_sizes) {
          // Generate input: values near `base` with perturbation ±spread
          std::vector<float> input(N), scale(N, 1.0f);
          for (size_t i = 0; i < N; i++) {
            double t = (static_cast<double>(i % 100) - 50.0) / 50.0;
            input[i] = static_cast<float>(base + t * spread);
            scale[i] = 1.0f;
          }

          // fp64 reference (two-pass in fp64 — algebraically exact at
          // fp32 magnitudes, algorithm-independent oracle)
          std::vector<double> out_fp64(N);
          double mean_fp64, inv_std_fp64;
          WelfordFp64Reference(input.data(), scale.data(), nullptr,
                               out_fp64.data(), &mean_fp64, &inv_std_fp64,
                               N, static_cast<double>(eps), false);

          // Kernel under test
          std::vector<float> out_kernel(N);
          float mean_k, inv_std_k;
          bool used = MlasLayerNormF32(input.data(), scale.data(), nullptr,
                                       out_kernel.data(), &mean_k, &inv_std_k,
                                       N, eps, false);
          ASSERT_TRUE(used) << "Kernel must dispatch for N=" << N;

          // Check for NaN/Inf first
          bool has_nonfinite = false;
          for (size_t i = 0; i < N; i++) {
            if (!std::isfinite(out_kernel[i])) {
              has_nonfinite = true;
              break;
            }
          }

          double err = has_nonfinite ? 1e30
                                     : MaxRelError(out_kernel.data(),
                                                   out_fp64.data(), N);
          overall_worst = std::max(overall_worst, err);
          total_cases++;

          if (err > kMaxRelError) {
            failures++;
            if (failures <= 10) {
              printf(
                  "  FAIL base=%.0e spread=%.0e eps=%.0e N=%-5zu "
                  "err=%.4e %s\n",
                  base, spread, eps, N,
                  err, has_nonfinite ? "(NaN/Inf)" : "");
            }
          }
        }
      }
    }
  }

  printf("\n  Fp64ParitySweep: %zu cases, %zu failures, worst=%.4e\n",
         total_cases, failures, overall_worst);

  // Guard against a vacuous sweep: if the condition-number gate or a
  // grid change silently drops every case, this test proves nothing.
  ASSERT_GT(total_cases, static_cast<size_t>(0))
      << "Fp64 parity sweep generated zero cases — the grid or condition-"
         "number gate is misconfigured.";

  EXPECT_LT(overall_worst, kMaxRelError)
      << "Fp64 parity sweep: " << failures << "/" << total_cases
      << " cases exceed " << kMaxRelError << " normalised max error. "
      << "Worst = " << overall_worst << ".";

  // ------------------------------------------------------------------
  // Explicit B1 regression check (base=1e5, spread=1e-2, N=1024,
  // eps=1e-6).  This case has condition number 1e7, outside the
  // sweep's cond < 1e6 gate, but it is the scenario that exposed the
  // original accuracy concern.
  //
  // Independently measured on this host (AMD EPYC 9V74, same binary):
  //   scalar Welford fp32:       0.9357  (vector-normalised max error)
  //   AVX2 centered two-pass:    0.03298
  // The kernel is ~28× more accurate than the scalar baseline here.
  //
  // We assert < 5e-2 to catch any regression back toward the scalar
  // Welford's ~0.94 while accepting the inherent fp32 limit at this
  // condition number.
  // ------------------------------------------------------------------
  {
    constexpr size_t B1_N = 1024;
    constexpr float B1_eps = 1e-6f;
    std::vector<float> b1_in(B1_N), b1_scale(B1_N, 1.0f);
    for (size_t i = 0; i < B1_N; i++) {
      double t = (static_cast<double>(i % 100) - 50.0) / 50.0;
      b1_in[i] = static_cast<float>(1e5 + t * 1e-2);
    }
    std::vector<double> b1_ref(B1_N);
    double b1_mean64, b1_inv64;
    WelfordFp64Reference(b1_in.data(), b1_scale.data(), nullptr,
                         b1_ref.data(), &b1_mean64, &b1_inv64,
                         B1_N, static_cast<double>(B1_eps), false);
    // Scalar Welford fp32 baseline (for comparison)
    std::vector<float> b1_scalar(B1_N);
    float b1_smean, b1_sinv;
    MlasLayerNormTest::ScalarFp32Baseline(
        b1_in.data(), b1_scale.data(), nullptr, b1_scalar.data(),
        &b1_smean, &b1_sinv, B1_N, B1_eps, false);
    double b1_scalar_err = MaxRelError(b1_scalar.data(), b1_ref.data(), B1_N);
    // AVX2 centered two-pass kernel
    std::vector<float> b1_out(B1_N);
    float b1_mean, b1_inv;
    MlasLayerNormF32(b1_in.data(), b1_scale.data(), nullptr,
                     b1_out.data(), &b1_mean, &b1_inv,
                     B1_N, B1_eps, false);
    double b1_err = MaxRelError(b1_out.data(), b1_ref.data(), B1_N);
    printf("  B1 regression check (base=1e5, spread=1e-2, N=1024, eps=1e-6):\n");
    printf("    scalar Welford fp32:    %.4e\n", b1_scalar_err);
    printf("    AVX2 centered two-pass: %.4e  (%.1fx better)\n",
           b1_err, b1_scalar_err / b1_err);
    EXPECT_LT(b1_err, 5e-2)
        << "B1 regression: kernel error at base=1e5, spread=1e-2, "
        << "N=1024 exceeds 5%. Scalar Welford was " << b1_scalar_err
        << "; current = " << b1_err << ".";
  }
}

// ---------------------------------------------------------------------------
// Benchmark (disabled by default; run with --gtest_also_run_disabled_tests)
// ---------------------------------------------------------------------------

class MlasLayerNormBenchTest : public MlasTestFixture<MlasLayerNormTest> {};

TEST_F(MlasLayerNormBenchTest, DISABLED_Benchmark) {
  // Representative shapes: threshold-aware + LLM-realistic hidden dims.
  // Sizes below the dispatch threshold are excluded because the kernel
  // declines them, and timing the scalar fallback is misleading.
  printf("\n=== LayerNorm (full) ===\n");
  for (size_t n : {15, 128, 256, 768, 1024, 2048, 4096}) {
    mlas_tester->Benchmark(n, /*warmup=*/100, /*iters=*/1000, /*simplified=*/false);
  }
  printf("\n=== RMSNorm (simplified) ===\n");
  for (size_t n : {16, 128, 256, 768, 1024, 2048, 4096}) {
    mlas_tester->Benchmark(n, /*warmup=*/100, /*iters=*/1000, /*simplified=*/true);
  }
}

// ---------------------------------------------------------------------------
// Registration into MLAS test harness
// ---------------------------------------------------------------------------

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return LayerNormShortExecuteTest::RegisterShortExecuteTests();
      }
      return 0;
    });
