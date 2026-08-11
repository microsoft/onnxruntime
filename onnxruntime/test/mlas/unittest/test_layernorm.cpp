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
    floor. The AVX2 kernel uses FMA contractions producing different rounding
    than the scalar fp64 reference. For small NormSize, the variance is near
    zero and 1/sqrt(var+eps) amplifies FMA rounding differences. The worst
    case observed is ~0.02% relative (NormSize=1, inv_stddev=316). Upstream
    CloseEnough uses rel_tol=0.005; we match that convention exactly.

--*/

#include "test_util.h"
#include "mlas.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------
// fp64-accumulated scalar reference (not dependent on MLAS)
// ---------------------------------------------------------------------------

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
  // Core test: numeric parity with reachability assertion.
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

    // REACHABILITY: the kernel MUST have dispatched on AVX2 hardware.
    // A silent fallback to scalar (used==false) is a test failure, not a skip.
    ASSERT_TRUE(used)
        << "REACHABILITY FAILURE: MlasLayerNormF32 returned false, meaning no "
           "optimized kernel dispatched. On AVX2 hardware the AVX2 LayerNorm "
           "kernel must be registered in platform.cpp. This is NOT a skip.";

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
    ASSERT_TRUE(near_enough(mean_mlas, mean_ref))
        << "mean mismatch got=" << mean_mlas << " ref=" << mean_ref;
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
    ASSERT_TRUE(used) << "Kernel must dispatch";

    // Zero-variance: all inputs equal, so (x - mean) should be ~0 but FMA
    // contraction in the AVX2 kernel may produce small nonzero residuals
    // (up to ~1.3e-4 observed). Use a wider absolute floor for this case.
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

  // Edge case: denormals
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
    ASSERT_TRUE(used);

    for (size_t i = 0; i < norm_size; i++) {
      ASSERT_TRUE(std::isfinite(output_mlas[i]))
          << "Non-finite on denormal input at [" << i << "]";
    }
  }

  // Edge case: large magnitudes
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
    ASSERT_TRUE(used);

    // NaN in → NaN out for both paths
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
  // that runs when MlasLayerNormF32() returns false on x86-64 prior to
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
      size_t norm_size,
      float epsilon,
      bool simplified) {
    float mean = 0.0f;
    float std_dev = 0.0f;

    if (simplified) {
      // RMSNorm: sum of squares, single pass
      float sum_sq = 0.0f;
      for (size_t h = 0; h < norm_size; h++) {
        output[h] = input[h];
        sum_sq += input[h] * input[h];
      }
      std_dev = sqrtf(sum_sq / static_cast<float>(norm_size) + epsilon);
    } else {
      // Welford's online algorithm — matches layer_norm_impl.cc exactly
      float M2 = 0.0f;
      for (size_t h = 0; h < norm_size; h++) {
        output[h] = input[h];
        float delta = input[h] - mean;
        mean += delta / static_cast<float>(h + 1);
        float delta2 = input[h] - mean;
        M2 += delta * delta2;
      }
      std_dev = sqrtf(M2 / static_cast<float>(norm_size) + epsilon);
    }

    float inv_denom = 1.0f / std_dev;
    for (size_t h = 0; h < norm_size; h++) {
      if (simplified) {
        output[h] = output[h] * inv_denom * scale[h];
      } else if (bias == nullptr) {
        output[h] = (output[h] - mean) * inv_denom * scale[h];
      } else {
        output[h] = (output[h] - mean) * inv_denom * scale[h] + bias[h];
      }
    }

    if (mean_out != nullptr) *mean_out = mean;
    if (inv_std_out != nullptr) *inv_std_out = inv_denom;
  }

  // Benchmark: AVX2 kernel vs true scalar fp32 baseline
  void Benchmark(size_t norm_size, size_t warmup, size_t iters, bool simplified) {
    std::vector<float> input(norm_size);
    std::vector<float> scale(norm_size);
    std::vector<float> output(norm_size);
    float mean_out, inv_std_out;

    for (size_t i = 0; i < norm_size; i++) {
      input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
      scale[i] = 1.0f + (static_cast<float>(i % 31) - 15.0f) * 0.001f;
    }

    // Warmup + measure: AVX2 kernel
    for (size_t i = 0; i < warmup; i++) {
      MlasLayerNormF32(input.data(), scale.data(), nullptr,
                       output.data(), &mean_out, &inv_std_out,
                       norm_size, 1e-5f, simplified);
    }
    std::vector<double> kernel_us(iters);
    for (size_t i = 0; i < iters; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      MlasLayerNormF32(input.data(), scale.data(), nullptr,
                       output.data(), &mean_out, &inv_std_out,
                       norm_size, 1e-5f, simplified);
      auto t1 = std::chrono::high_resolution_clock::now();
      kernel_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // Warmup + measure: scalar fp32 baseline (the actual code being replaced)
    for (size_t i = 0; i < warmup; i++) {
      ScalarFp32Baseline(input.data(), scale.data(), nullptr,
                         output.data(), &mean_out, &inv_std_out,
                         norm_size, 1e-5f, simplified);
    }
    std::vector<double> scalar_us(iters);
    for (size_t i = 0; i < iters; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      ScalarFp32Baseline(input.data(), scale.data(), nullptr,
                         output.data(), &mean_out, &inv_std_out,
                         norm_size, 1e-5f, simplified);
      auto t1 = std::chrono::high_resolution_clock::now();
      scalar_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // Also measure the fp64 reference for context (the independent oracle baseline)
    for (size_t i = 0; i < warmup; i++) {
      ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                         output.data(), &mean_out, &inv_std_out,
                         norm_size, 1e-5f, simplified);
    }
    std::vector<double> fp64_us(iters);
    for (size_t i = 0; i < iters; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      ReferenceLayerNorm(input.data(), scale.data(), nullptr,
                         output.data(), &mean_out, &inv_std_out,
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
// Benchmark (disabled by default; run with --gtest_also_run_disabled_tests)
// ---------------------------------------------------------------------------

class MlasLayerNormBenchTest : public MlasTestFixture<MlasLayerNormTest> {};

TEST_F(MlasLayerNormBenchTest, DISABLED_Benchmark) {
  // Representative shapes: small/tail sizes + LLM-realistic hidden dims
  printf("\n=== LayerNorm (full) ===\n");
  for (size_t n : {7, 15, 128, 256, 768, 1024, 2048, 4096}) {
    mlas_tester->Benchmark(n, /*warmup=*/100, /*iters=*/1000, /*simplified=*/false);
  }
  printf("\n=== RMSNorm (simplified) ===\n");
  for (size_t n : {7, 15, 128, 256, 768, 1024, 2048, 4096}) {
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
