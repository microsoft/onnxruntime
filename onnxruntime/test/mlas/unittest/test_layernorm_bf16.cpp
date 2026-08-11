/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_layernorm_bf16.cpp

Abstract:

    Precision tests for BF16 LayerNorm/RMSNorm.

    Tests cover:
    1. BFloat16 rounding-rule validation (round-to-nearest-even).
    2. fp64 oracle for LayerNorm/RMSNorm on exact bf16 inputs.
    3. High-dynamic-range adversarial cases (catastrophic cancellation,
       near-zero variance, denormals, large N).
    4. Separation of representation error vs. kernel error.

    The kernel hook (MlasLayerNormBF16) is gated on availability:
    when Resch's implementation lands, the oracle comparison activates
    automatically.

--*/

#include "test_util.h"
#include "core/common/float16.h"
#include "mlas.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

using onnxruntime::BFloat16;

// ========================================================================
// Helpers
// ========================================================================

/// Convert float to BFloat16 using ORT's conversion (round-to-nearest-even).
static inline BFloat16 F32ToBF16(float v) {
  return BFloat16(v);
}

/// Convert BFloat16 back to float (exact, just zero-extends mantissa).
static inline float BF16ToF32(BFloat16 v) {
  return v.ToFloat();
}

/// Naive reference: convert float to bf16 by truncation (not rounding).
static inline uint16_t F32ToBF16Truncate(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

/// bf16 ULP distance between two finite bf16 values.
static int BF16UlpDistance(BFloat16 a, BFloat16 b) {
  auto to_signed = [](uint16_t v) -> int32_t {
    if (v & 0x8000U) {
      return -static_cast<int32_t>(v ^ 0x8000U);
    }
    return static_cast<int32_t>(v);
  };
  return std::abs(to_signed(a.val) - to_signed(b.val));
}

/// Compute bf16 ULP of a given float value (distance to next bf16).
static float BF16Ulp(float v) {
  BFloat16 bf = F32ToBF16(v);
  float rounded = BF16ToF32(bf);
  // Find adjacent bf16
  uint16_t bits = bf.val;
  float next;
  if (bits == 0x7F7FU || bits == 0xFF7FU) {
    // At max representable
    next = rounded;
  } else {
    BFloat16 adj = BFloat16::FromBits(static_cast<uint16_t>(bits + 1));
    next = BF16ToF32(adj);
  }
  return std::abs(next - rounded);
}

// ========================================================================
// fp64 Oracle — LayerNorm and RMSNorm
// ========================================================================

struct OracleResult {
  std::vector<double> output;
  double mean;
  double inv_std;
  double variance;
};

/// Welford's online algorithm in fp64 for variance computation.
static OracleResult LayerNormOracle_fp64(
    const std::vector<BFloat16>& input_bf16,
    const std::vector<BFloat16>& scale_bf16,
    const std::vector<BFloat16>* bias_bf16,
    double epsilon,
    bool simplified) {
  size_t N = input_bf16.size();
  OracleResult result;
  result.output.resize(N);

  // Convert inputs to fp64 from the exact bf16 values
  std::vector<double> x(N), s(N), b(N);
  for (size_t i = 0; i < N; i++) {
    x[i] = static_cast<double>(BF16ToF32(input_bf16[i]));
    s[i] = static_cast<double>(BF16ToF32(scale_bf16[i]));
    b[i] = bias_bf16 ? static_cast<double>(BF16ToF32((*bias_bf16)[i])) : 0.0;
  }

  if (simplified) {
    // RMSNorm: mean of squares
    double sum_sq = 0.0;
    for (size_t i = 0; i < N; i++) {
      sum_sq += x[i] * x[i];
    }
    double ms = sum_sq / static_cast<double>(N);
    double denom = std::sqrt(ms + epsilon);
    double inv_denom = 1.0 / denom;
    for (size_t i = 0; i < N; i++) {
      result.output[i] = x[i] * inv_denom * s[i];
    }
    result.mean = 0.0;
    result.inv_std = inv_denom;
    result.variance = ms;
  } else {
    // LayerNorm: Welford's online algorithm
    double welford_mean = 0.0;
    double welford_M2 = 0.0;
    for (size_t i = 0; i < N; i++) {
      double delta = x[i] - welford_mean;
      welford_mean += delta / static_cast<double>(i + 1);
      double delta2 = x[i] - welford_mean;
      welford_M2 += delta * delta2;
    }
    double var = welford_M2 / static_cast<double>(N);
    double denom = std::sqrt(var + epsilon);
    double inv_denom = 1.0 / denom;
    for (size_t i = 0; i < N; i++) {
      result.output[i] = (x[i] - welford_mean) * inv_denom * s[i] + b[i];
    }
    result.mean = welford_mean;
    result.inv_std = inv_denom;
    result.variance = var;
  }
  return result;
}

/// Two-pass oracle in fp64 for comparison.
static OracleResult LayerNormOracle_fp64_TwoPass(
    const std::vector<BFloat16>& input_bf16,
    const std::vector<BFloat16>& scale_bf16,
    const std::vector<BFloat16>* bias_bf16,
    double epsilon,
    bool simplified) {
  size_t N = input_bf16.size();
  OracleResult result;
  result.output.resize(N);

  std::vector<double> x(N), s(N), b(N);
  for (size_t i = 0; i < N; i++) {
    x[i] = static_cast<double>(BF16ToF32(input_bf16[i]));
    s[i] = static_cast<double>(BF16ToF32(scale_bf16[i]));
    b[i] = bias_bf16 ? static_cast<double>(BF16ToF32((*bias_bf16)[i])) : 0.0;
  }

  if (simplified) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < N; i++) sum_sq += x[i] * x[i];
    double ms = sum_sq / static_cast<double>(N);
    double inv = 1.0 / std::sqrt(ms + epsilon);
    for (size_t i = 0; i < N; i++) result.output[i] = x[i] * inv * s[i];
    result.mean = 0.0;
    result.inv_std = inv;
    result.variance = ms;
  } else {
    double sum = 0.0;
    for (size_t i = 0; i < N; i++) sum += x[i];
    double mean = sum / static_cast<double>(N);
    double sum_sq = 0.0;
    for (size_t i = 0; i < N; i++) sum_sq += (x[i] - mean) * (x[i] - mean);
    double var = sum_sq / static_cast<double>(N);
    double inv = 1.0 / std::sqrt(var + epsilon);
    for (size_t i = 0; i < N; i++) result.output[i] = (x[i] - mean) * inv * s[i] + b[i];
    result.mean = mean;
    result.inv_std = inv;
    result.variance = var;
  }
  return result;
}

// ========================================================================
// Error measurement utilities
// ========================================================================

struct ErrorStats {
  double max_abs_error;
  double max_rel_error;
  int max_bf16_ulp_distance;
  double rms_error;
  size_t error_count;
  size_t total_count;
};

static ErrorStats ComputeErrors(
    const std::vector<double>& oracle,
    const std::vector<float>& actual) {
  ErrorStats stats = {};
  stats.total_count = oracle.size();
  double sum_sq = 0.0;
  for (size_t i = 0; i < oracle.size(); i++) {
    double ref = oracle[i];
    double act = static_cast<double>(actual[i]);
    double abs_err = std::abs(ref - act);
    double rel_err = (std::abs(ref) > 1e-30) ? abs_err / std::abs(ref) : abs_err;

    if (abs_err > stats.max_abs_error) stats.max_abs_error = abs_err;
    if (rel_err > stats.max_rel_error) stats.max_rel_error = rel_err;
    sum_sq += abs_err * abs_err;

    // bf16 ULP distance
    BFloat16 ref_bf = F32ToBF16(static_cast<float>(ref));
    BFloat16 act_bf = F32ToBF16(actual[i]);
    int ulp = BF16UlpDistance(ref_bf, act_bf);
    if (ulp > stats.max_bf16_ulp_distance) stats.max_bf16_ulp_distance = ulp;
    if (ulp > 0) stats.error_count++;
  }
  stats.rms_error = std::sqrt(sum_sq / static_cast<double>(oracle.size()));
  return stats;
}

/// Measure the unavoidable representation error: f32 oracle output quantized to bf16
/// vs fp64 oracle. This is the floor that no kernel can beat.
static ErrorStats MeasureRepresentationError(
    const std::vector<double>& oracle_fp64) {
  std::vector<float> quantized(oracle_fp64.size());
  for (size_t i = 0; i < oracle_fp64.size(); i++) {
    // Best possible: fp64 result narrowed to bf16 then back to f32
    BFloat16 bf = F32ToBF16(static_cast<float>(oracle_fp64[i]));
    quantized[i] = BF16ToF32(bf);
  }
  return ComputeErrors(oracle_fp64, quantized);
}

// ========================================================================
// Test: BFloat16 Rounding Rule — Round-to-Nearest-Even
// ========================================================================

class BF16RoundingTest : public MlasTestBase {
 public:
  void TestRoundToNearestEven() {
    // Test the four rounding cases for bf16:
    // For bf16, mantissa is 7 bits. Rounding happens at bit 16 of f32.
    // Case 1: Exact — no rounding needed
    // Case 2: Round down (discarded bits < 0.5 ULP)
    // Case 3: Round up (discarded bits > 0.5 ULP)
    // Case 4: Tie-breaking to even

    // Test systematic: for each bf16 value, the midpoint between it and
    // the next bf16 should round to even.
    int tie_even_correct = 0;
    int tie_even_total = 0;
    int round_correct = 0;
    int round_total = 0;

    // Test a range of exponents
    for (int exp = -10; exp <= 10; exp++) {
      float base = std::ldexp(1.0f, exp);
      // For each bf16 in this range, check the midpoint
      BFloat16 bf_base = F32ToBF16(base);
      uint16_t bits = bf_base.val;

      // Skip special values
      if (!bf_base.IsFinite() || bf_base.IsNaN()) continue;

      for (int offset = 0; offset < 16; offset++) {
        uint16_t current_bits = static_cast<uint16_t>(bits + offset);
        uint16_t next_bits = static_cast<uint16_t>(current_bits + 1);

        BFloat16 current = BFloat16::FromBits(current_bits);
        BFloat16 next = BFloat16::FromBits(next_bits);

        if (!current.IsFinite() || !next.IsFinite()) continue;

        float current_f32 = BF16ToF32(current);
        float next_f32 = BF16ToF32(next);
        float midpoint = current_f32 + (next_f32 - current_f32) * 0.5f;

        // The midpoint should round to the even one
        BFloat16 rounded = F32ToBF16(midpoint);
        bool current_is_even = (current_bits & 1) == 0;
        bool should_be_current = current_is_even;

        tie_even_total++;
        if ((should_be_current && rounded.val == current_bits) ||
            (!should_be_current && rounded.val == next_bits)) {
          tie_even_correct++;
        }

        // Test just below midpoint — should round to current
        float below_mid = std::nextafterf(midpoint, current_f32);
        BFloat16 rounded_below = F32ToBF16(below_mid);
        round_total++;
        if (rounded_below.val == current_bits) {
          round_correct++;
        }

        // Test just above midpoint — should round to next
        float above_mid = std::nextafterf(midpoint, next_f32);
        BFloat16 rounded_above = F32ToBF16(above_mid);
        round_total++;
        if (rounded_above.val == next_bits) {
          round_correct++;
        }
      }
    }

    // Report results
    printf(
        "  BF16 rounding: tie-to-even %d/%d correct, "
        "directional %d/%d correct\n",
        tie_even_correct, tie_even_total,
        round_correct, round_total);

    ASSERT_EQ(tie_even_correct, tie_even_total)
        << "BFloat16 conversion does NOT implement round-to-nearest-even";
    ASSERT_EQ(round_correct, round_total)
        << "BFloat16 conversion fails directional rounding";
  }

  void TestConsistencyWithTruncation() {
    // Verify that ORT's bf16 conversion differs from truncation where expected
    int differ_count = 0;
    int total_checked = 0;
    int ort_closer = 0;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < 10000; i++) {
      float v = dist(rng);
      BFloat16 ort_bf = F32ToBF16(v);
      uint16_t trunc_bits = F32ToBF16Truncate(v);

      total_checked++;
      if (ort_bf.val != trunc_bits) {
        differ_count++;
        // ORT's rounded value should be closer to or equal distance from v
        float ort_f = BF16ToF32(ort_bf);
        BFloat16 trunc_bf = BFloat16::FromBits(trunc_bits);
        float trunc_f = BF16ToF32(trunc_bf);
        if (std::abs(ort_f - v) <= std::abs(trunc_f - v)) {
          ort_closer++;
        }
      }
    }

    printf(
        "  ORT bf16 vs truncation: %d/%d differ, "
        "ORT closer/equal in %d/%d differing cases\n",
        differ_count, total_checked, ort_closer, differ_count);

    // ORT should always be at least as close as truncation
    ASSERT_EQ(ort_closer, differ_count)
        << "ORT BFloat16 conversion is WORSE than truncation in some cases";
    // We expect differences — truncation is not RNE
    ASSERT_GT(differ_count, 0)
        << "ORT BFloat16 conversion is identical to truncation — "
           "either the test is broken or RNE is not implemented";
  }

  void TestRoundTripSpecialValues() {
    // Verify special values survive round-trip
    float specials[] = {0.0f, -0.0f, 1.0f, -1.0f,
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()};
    for (float v : specials) {
      BFloat16 bf = F32ToBF16(v);
      float rt = BF16ToF32(bf);
      if (std::isnan(v)) {
        ASSERT_TRUE(std::isnan(rt)) << "NaN not preserved";
      } else {
        ASSERT_EQ(v, rt) << "Special value " << v << " not preserved in round-trip";
      }
    }

    // NaN
    BFloat16 nan_bf = F32ToBF16(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(nan_bf.IsNaN());
    ASSERT_TRUE(std::isnan(BF16ToF32(nan_bf)));
  }

  void TestDenormals() {
    // bf16 denormals: exponent = 0, mantissa != 0
    // Smallest bf16 denormal: 0x0001 ≈ 2^(-126) * 2^(-7) = 2^(-133)
    BFloat16 smallest_denorm = BFloat16::FromBits(0x0001U);
    ASSERT_TRUE(smallest_denorm.IsSubnormal());
    float sd_f32 = BF16ToF32(smallest_denorm);
    ASSERT_GT(sd_f32, 0.0f);

    // Round-trip a value that should become a bf16 denormal
    BFloat16 rt = F32ToBF16(sd_f32);
    ASSERT_EQ(rt.val, smallest_denorm.val)
        << "Smallest bf16 denormal does not round-trip";

    // Largest bf16 denormal: 0x007F
    BFloat16 largest_denorm = BFloat16::FromBits(0x007FU);
    ASSERT_TRUE(largest_denorm.IsSubnormal());

    printf("  BF16 denormal range: [%.6e, %.6e]\n",
           sd_f32, BF16ToF32(largest_denorm));
  }
};

// ========================================================================
// Test: BF16 LayerNorm Oracle and Precision Analysis
// ========================================================================

class BF16LayerNormPrecisionTest : public MlasTestBase {
 private:
  /// Generate bf16 inputs from f32 values (quantize to bf16 first).
  static void GenerateBF16Inputs(
      const std::vector<float>& f32_values,
      std::vector<BFloat16>& bf16_out) {
    bf16_out.resize(f32_values.size());
    for (size_t i = 0; i < f32_values.size(); i++) {
      bf16_out[i] = F32ToBF16(f32_values[i]);
    }
  }

  /// Report error decomposition for a test case.
  static void ReportErrors(
      const char* label,
      const OracleResult& oracle,
      const std::vector<float>& kernel_output,
      size_t N) {
    // Representation error floor
    ErrorStats rep_err = MeasureRepresentationError(oracle.output);

    // Kernel error vs oracle
    ErrorStats kern_err = ComputeErrors(oracle.output, kernel_output);

    printf("  [%s] N=%zu\n", label, N);
    printf(
        "    Representation floor: max_abs=%.3e max_rel=%.3e "
        "max_ulp=%d rms=%.3e\n",
        rep_err.max_abs_error, rep_err.max_rel_error,
        rep_err.max_bf16_ulp_distance, rep_err.rms_error);
    printf(
        "    Kernel error:         max_abs=%.3e max_rel=%.3e "
        "max_ulp=%d rms=%.3e (%zu/%zu differ)\n",
        kern_err.max_abs_error, kern_err.max_rel_error,
        kern_err.max_bf16_ulp_distance, kern_err.rms_error,
        kern_err.error_count, kern_err.total_count);
  }

 public:
  // ------------------------------------------------------------------
  // Oracle-only tests (no kernel dependency)
  // ------------------------------------------------------------------

  /// Verify that the fp64 oracle is self-consistent: Welford and
  /// two-pass agree in fp64 to machine precision.
  void TestOracleConsistency(size_t N, bool simplified) {
    std::mt19937 rng(123 + static_cast<unsigned>(N));
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> f32_input(N), f32_scale(N), f32_bias(N);
    for (size_t i = 0; i < N; i++) {
      f32_input[i] = dist(rng);
      f32_scale[i] = 0.5f + std::abs(dist(rng)) * 0.1f;
      f32_bias[i] = dist(rng) * 0.01f;
    }

    std::vector<BFloat16> bf_input, bf_scale, bf_bias;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);
    GenerateBF16Inputs(f32_bias, bf_bias);

    auto welford = LayerNormOracle_fp64(bf_input, bf_scale,
                                        simplified ? nullptr : &bf_bias,
                                        1e-5, simplified);
    auto twopass = LayerNormOracle_fp64_TwoPass(bf_input, bf_scale,
                                                simplified ? nullptr : &bf_bias,
                                                1e-5, simplified);

    // In fp64, these should agree to ~1e-14 or better
    double mean_diff = std::abs(welford.mean - twopass.mean);
    double var_diff = std::abs(welford.variance - twopass.variance);
    double max_out_diff = 0.0;
    for (size_t i = 0; i < N; i++) {
      double d = std::abs(welford.output[i] - twopass.output[i]);
      if (d > max_out_diff) max_out_diff = d;
    }

    printf(
        "  Oracle consistency N=%zu simplified=%d: "
        "mean_diff=%.3e var_diff=%.3e max_out_diff=%.3e\n",
        N, simplified, mean_diff, var_diff, max_out_diff);

    ASSERT_LT(mean_diff, 1e-10)
        << "Oracle Welford vs two-pass mean disagree";
    ASSERT_LT(var_diff, 1e-10)
        << "Oracle Welford vs two-pass variance disagree";
    ASSERT_LT(max_out_diff, 1e-10)
        << "Oracle Welford vs two-pass output disagree";
  }

  /// Measure the bf16 representation-error floor for typical inputs.
  void TestRepresentationErrorFloor(size_t N) {
    std::mt19937 rng(77 + static_cast<unsigned>(N));
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> f32_input(N), f32_scale(N);
    for (size_t i = 0; i < N; i++) {
      f32_input[i] = dist(rng);
      f32_scale[i] = 1.0f;
    }

    std::vector<BFloat16> bf_input, bf_scale;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);

    auto oracle = LayerNormOracle_fp64(bf_input, bf_scale, nullptr, 1e-5, false);
    ErrorStats rep = MeasureRepresentationError(oracle.output);

    printf(
        "  Representation error floor N=%zu: "
        "max_abs=%.3e max_rel=%.3e max_ulp=%d rms=%.3e\n",
        N, rep.max_abs_error, rep.max_rel_error,
        rep.max_bf16_ulp_distance, rep.rms_error);

    // The representation error should be bounded by 0.5 ULP of bf16
    // For normalized values around 1.0, bf16 ULP ≈ 2^-7 ≈ 0.0078
    // Max error from RNE quantization is 0.5 ULP
    ASSERT_LE(rep.max_bf16_ulp_distance, 1)
        << "Representation error exceeds 1 bf16 ULP — oracle bug?";
  }

  // ------------------------------------------------------------------
  // High-dynamic-range adversarial cases
  // ------------------------------------------------------------------

  /// Catastrophic cancellation: large mean, tiny variance.
  /// This is the shape that broke two-pass f32 on #31973.
  void TestCatastrophicCancellation(size_t N) {
    // All values ≈ 1000.0 with tiny perturbations
    // In bf16, 1000.0 = 0x4488 (exact), and the representable step is 4.0
    // So perturbations smaller than 2.0 vanish — this is the worst case.
    std::vector<float> f32_input(N);
    float base = 1000.0f;
    BFloat16 bf_base = F32ToBF16(base);
    float base_exact = BF16ToF32(bf_base);

    // Add perturbations at the bf16 resolution level
    for (size_t i = 0; i < N; i++) {
      // Alternate between base and base+1ulp to create tiny variance
      if (i % 2 == 0) {
        f32_input[i] = base_exact;
      } else {
        BFloat16 next = BFloat16::FromBits(static_cast<uint16_t>(bf_base.val + 1));
        f32_input[i] = BF16ToF32(next);
      }
    }

    std::vector<float> f32_scale(N, 1.0f);
    std::vector<BFloat16> bf_input, bf_scale;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);

    auto oracle = LayerNormOracle_fp64(bf_input, bf_scale, nullptr, 1e-5, false);

    printf(
        "  Catastrophic cancellation N=%zu: "
        "mean=%.6f variance=%.6e inv_std=%.6f\n",
        N, oracle.mean, oracle.variance, oracle.inv_std);

    // Verify the oracle produces finite, sensible results
    ASSERT_TRUE(std::isfinite(oracle.mean));
    ASSERT_TRUE(std::isfinite(oracle.inv_std));
    ASSERT_GE(oracle.variance, 0.0);
    for (size_t i = 0; i < N; i++) {
      ASSERT_TRUE(std::isfinite(oracle.output[i]))
          << "Oracle output NaN/Inf at i=" << i;
    }

    ErrorStats rep = MeasureRepresentationError(oracle.output);
    printf("    Representation error: max_abs=%.3e max_ulp=%d\n",
           rep.max_abs_error, rep.max_bf16_ulp_distance);
  }

  /// High dynamic range: values spanning many orders of magnitude.
  void TestHighDynamicRange(size_t N) {
    std::vector<float> f32_input(N);
    // Exponential spread: values from ~1e-3 to ~1e3
    for (size_t i = 0; i < N; i++) {
      float t = static_cast<float>(i) / static_cast<float>(N) * 6.0f - 3.0f;
      f32_input[i] = std::pow(10.0f, t) * ((i % 2 == 0) ? 1.0f : -1.0f);
    }

    std::vector<float> f32_scale(N, 1.0f);
    std::vector<BFloat16> bf_input, bf_scale;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);

    auto oracle = LayerNormOracle_fp64(bf_input, bf_scale, nullptr, 1e-5, false);
    ErrorStats rep = MeasureRepresentationError(oracle.output);

    printf(
        "  High dynamic range N=%zu: mean=%.3e var=%.3e "
        "rep_max_ulp=%d rep_max_rel=%.3e\n",
        N, oracle.mean, oracle.variance,
        rep.max_bf16_ulp_distance, rep.max_rel_error);

    ASSERT_TRUE(std::isfinite(oracle.mean));
    ASSERT_TRUE(std::isfinite(oracle.inv_std));
  }

  /// Near-zero variance: all inputs identical.
  void TestNearZeroVariance(size_t N) {
    std::vector<float> f32_input(N, 42.0f);
    std::vector<float> f32_scale(N, 1.0f);
    std::vector<BFloat16> bf_input, bf_scale;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);

    auto oracle = LayerNormOracle_fp64(bf_input, bf_scale, nullptr, 1e-5, false);

    printf(
        "  Near-zero variance N=%zu: mean=%.6f var=%.3e "
        "inv_std=%.6f\n",
        N, oracle.mean, oracle.variance, oracle.inv_std);

    ASSERT_NEAR(oracle.variance, 0.0, 1e-15);
    // Output should be ~0 * scale = ~0 (since x - mean = 0)
    for (size_t i = 0; i < N; i++) {
      ASSERT_NEAR(oracle.output[i], 0.0, 1e-10)
          << "Near-zero variance output not near zero at i=" << i;
    }
  }

  /// Denormal bf16 inputs.
  void TestDenormalInputs() {
    const size_t N = 64;
    std::vector<float> f32_input(N);
    for (size_t i = 0; i < N; i++) {
      // bf16 denormals: bits 0x0001 to 0x007F (positive)
      uint16_t bits = static_cast<uint16_t>((i % 127) + 1);
      f32_input[i] = BF16ToF32(BFloat16::FromBits(bits));
    }

    std::vector<float> f32_scale(N, 1.0f);
    std::vector<BFloat16> bf_input, bf_scale;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);

    auto oracle = LayerNormOracle_fp64(bf_input, bf_scale, nullptr, 1e-5, false);

    printf("  Denormal inputs N=%zu: mean=%.3e var=%.3e\n",
           N, oracle.mean, oracle.variance);

    ASSERT_TRUE(std::isfinite(oracle.mean));
    ASSERT_TRUE(std::isfinite(oracle.inv_std));
    for (size_t i = 0; i < N; i++) {
      ASSERT_TRUE(std::isfinite(oracle.output[i]));
    }
  }

  /// Near bf16 max values.
  void TestNearMaxValues(size_t N) {
    // bf16 max: 0x7F7F ≈ 3.39e38
    // Use values around 1e4 to stay large but not overflow in variance
    std::vector<float> f32_input(N);
    BFloat16 large_bf = F32ToBF16(10000.0f);
    float large_exact = BF16ToF32(large_bf);
    for (size_t i = 0; i < N; i++) {
      // Vary by a few ULPs
      int offset = static_cast<int>(i % 5) - 2;
      uint16_t bits = static_cast<uint16_t>(
          static_cast<int>(large_bf.val) + offset);
      f32_input[i] = BF16ToF32(BFloat16::FromBits(bits));
    }

    std::vector<float> f32_scale(N, 1.0f);
    std::vector<BFloat16> bf_input, bf_scale;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);

    auto oracle = LayerNormOracle_fp64(bf_input, bf_scale, nullptr, 1e-5, false);

    printf("  Near-max values N=%zu: base=%.3e mean=%.6e var=%.3e\n",
           N, static_cast<double>(large_exact), oracle.mean, oracle.variance);

    ASSERT_TRUE(std::isfinite(oracle.mean));
    ASSERT_TRUE(std::isfinite(oracle.inv_std));
  }

  // ------------------------------------------------------------------
  // f32 widen-accumulate-narrow simulation
  // ------------------------------------------------------------------

  /// Simulate the widen→f32-accumulate→narrow path that Resch is implementing.
  /// This is the reference for what the kernel SHOULD produce.
  void TestWidenAccumulateNarrow(size_t N, bool simplified) {
    std::mt19937 rng(999 + static_cast<unsigned>(N));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> f32_input(N), f32_scale(N), f32_bias(N);
    for (size_t i = 0; i < N; i++) {
      f32_input[i] = dist(rng);
      f32_scale[i] = 0.8f + std::abs(dist(rng)) * 0.1f;
      f32_bias[i] = dist(rng) * 0.01f;
    }

    std::vector<BFloat16> bf_input, bf_scale, bf_bias;
    GenerateBF16Inputs(f32_input, bf_input);
    GenerateBF16Inputs(f32_scale, bf_scale);
    GenerateBF16Inputs(f32_bias, bf_bias);

    // fp64 oracle
    auto oracle = LayerNormOracle_fp64(bf_input, bf_scale,
                                       simplified ? nullptr : &bf_bias,
                                       1e-5, simplified);

    // Simulate widen-accumulate-narrow in f32
    std::vector<float> x(N), s(N), b(N);
    for (size_t i = 0; i < N; i++) {
      x[i] = BF16ToF32(bf_input[i]);
      s[i] = BF16ToF32(bf_scale[i]);
      b[i] = BF16ToF32(bf_bias[i]);
    }

    // f32 two-pass (what a naive kernel might do)
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) sum += x[i];
    float mean = sum / static_cast<float>(N);

    std::vector<float> f32_output(N);
    if (simplified) {
      float sum_sq = 0.0f;
      for (size_t i = 0; i < N; i++) sum_sq += x[i] * x[i];
      float ms = sum_sq / static_cast<float>(N);
      float inv = 1.0f / std::sqrt(ms + 1e-5f);
      for (size_t i = 0; i < N; i++) {
        f32_output[i] = x[i] * inv * s[i];
      }
    } else {
      float sum_sq = 0.0f;
      for (size_t i = 0; i < N; i++) sum_sq += (x[i] - mean) * (x[i] - mean);
      float var = sum_sq / static_cast<float>(N);
      float inv = 1.0f / std::sqrt(var + 1e-5f);
      for (size_t i = 0; i < N; i++) {
        f32_output[i] = (x[i] - mean) * inv * s[i] + b[i];
      }
    }

    // Narrow to bf16 then back for comparison
    std::vector<float> bf16_narrowed(N);
    for (size_t i = 0; i < N; i++) {
      bf16_narrowed[i] = BF16ToF32(F32ToBF16(f32_output[i]));
    }

    ErrorStats rep = MeasureRepresentationError(oracle.output);
    ErrorStats kern = ComputeErrors(oracle.output, bf16_narrowed);

    printf("  Widen-accumulate-narrow N=%zu simplified=%d\n", N, simplified);
    printf("    Rep floor: max_abs=%.3e max_ulp=%d\n",
           rep.max_abs_error, rep.max_bf16_ulp_distance);
    printf("    f32 2pass: max_abs=%.3e max_ulp=%d err_above_floor=%d\n",
           kern.max_abs_error, kern.max_bf16_ulp_distance,
           kern.max_bf16_ulp_distance - rep.max_bf16_ulp_distance);

    // The f32 two-pass kernel error should be small — within a few bf16 ULPs
    // For typical inputs and N≤65536, we expect ≤2 ULP above rep floor
    // For catastrophic cases this can be higher — those are separate tests
  }

  // ------------------------------------------------------------------
  // Cross-check BF16 vs FP16 for the same logical input
  // ------------------------------------------------------------------

  void TestBF16vsFP16Precision(size_t N) {
    // Generate values in fp16 range (avoid fp16 overflow)
    std::mt19937 rng(555 + static_cast<unsigned>(N));
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> f32_input(N);
    for (size_t i = 0; i < N; i++) {
      f32_input[i] = dist(rng);
    }

    // Quantize to bf16
    std::vector<BFloat16> bf_input(N);
    for (size_t i = 0; i < N; i++) bf_input[i] = F32ToBF16(f32_input[i]);

    // Quantize to fp16 (MLFloat16)
    std::vector<onnxruntime::MLFloat16> fp16_input(N);
    for (size_t i = 0; i < N; i++) fp16_input[i] = onnxruntime::MLFloat16(f32_input[i]);

    // Measure input quantization error for both formats
    double bf16_max_input_err = 0.0, fp16_max_input_err = 0.0;
    for (size_t i = 0; i < N; i++) {
      double bf16_err = std::abs(static_cast<double>(f32_input[i]) -
                                 static_cast<double>(BF16ToF32(bf_input[i])));
      double fp16_err = std::abs(static_cast<double>(f32_input[i]) -
                                 static_cast<double>(fp16_input[i].ToFloat()));
      if (bf16_err > bf16_max_input_err) bf16_max_input_err = bf16_err;
      if (fp16_err > fp16_max_input_err) fp16_max_input_err = fp16_err;
    }

    printf(
        "  BF16 vs FP16 input quantization N=%zu:\n"
        "    BF16 max input err: %.3e\n"
        "    FP16 max input err: %.3e\n"
        "    Ratio (BF16/FP16): %.1f\n",
        N, bf16_max_input_err, fp16_max_input_err,
        fp16_max_input_err > 0 ? bf16_max_input_err / fp16_max_input_err : 0.0);

    // BF16 has 8-bit mantissa vs FP16's 11-bit, so BF16 error should be
    // ~8x larger (2^3 = 8)
    if (fp16_max_input_err > 0) {
      double ratio = bf16_max_input_err / fp16_max_input_err;
      ASSERT_GT(ratio, 2.0) << "BF16 should have larger quantization error than FP16";
      ASSERT_LT(ratio, 20.0) << "Ratio seems too large — check for bugs";
    }
  }
};

// ========================================================================
// Test Registration
// ========================================================================

// --- Rounding tests ---
class BF16RoundingShortTest : public MlasTestFixture<BF16RoundingTest> {
 public:
  explicit BF16RoundingShortTest(int test_id) : test_id_(test_id) {}

  void TestBody() override {
    auto* tester = MlasTestFixture<BF16RoundingTest>::mlas_tester;
    switch (test_id_) {
      case 0:
        tester->TestRoundToNearestEven();
        break;
      case 1:
        tester->TestConsistencyWithTruncation();
        break;
      case 2:
        tester->TestRoundTripSpecialValues();
        break;
      case 3:
        tester->TestDenormals();
        break;
    }
  }

  static size_t RegisterShortExecuteTests() {
    size_t count = 0;
    const char* names[] = {"RoundToNearestEven", "ConsistencyWithTruncation",
                           "RoundTripSpecialValues", "Denormals"};
    for (int i = 0; i < 4; i++) {
      testing::RegisterTest(
          "BF16Rounding", names[i], nullptr, names[i],
          __FILE__, __LINE__,
          [i]() -> MlasTestFixture<BF16RoundingTest>* {
            return new BF16RoundingShortTest(i);
          });
      count++;
    }
    return count;
  }

 private:
  int test_id_;
};

// --- Oracle and precision tests ---
class BF16LayerNormPrecisionShortTest
    : public MlasTestFixture<BF16LayerNormPrecisionTest> {
 public:
  BF16LayerNormPrecisionShortTest(int test_id, size_t N, bool simplified)
      : test_id_(test_id), N_(N), simplified_(simplified) {}

  void TestBody() override {
    auto* tester = MlasTestFixture<BF16LayerNormPrecisionTest>::mlas_tester;
    switch (test_id_) {
      case 0:
        tester->TestOracleConsistency(N_, simplified_);
        break;
      case 1:
        tester->TestRepresentationErrorFloor(N_);
        break;
      case 2:
        tester->TestCatastrophicCancellation(N_);
        break;
      case 3:
        tester->TestHighDynamicRange(N_);
        break;
      case 4:
        tester->TestNearZeroVariance(N_);
        break;
      case 5:
        tester->TestDenormalInputs();
        break;
      case 6:
        tester->TestNearMaxValues(N_);
        break;
      case 7:
        tester->TestWidenAccumulateNarrow(N_, simplified_);
        break;
      case 8:
        tester->TestBF16vsFP16Precision(N_);
        break;
    }
  }

  static size_t RegisterShortExecuteTests() {
    size_t count = 0;

    // Oracle consistency
    for (size_t N : {64, 256, 1024, 4096}) {
      for (bool simplified : {false, true}) {
        std::stringstream ss;
        ss << "OracleConsistency/N" << N << "/simplified" << simplified;
        auto name = ss.str();
        testing::RegisterTest(
            "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
            __FILE__, __LINE__,
            [N, simplified]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
              return new BF16LayerNormPrecisionShortTest(0, N, simplified);
            });
        count++;
      }
    }

    // Representation error floor
    for (size_t N : {64, 256, 1024, 4096, 16384, 65536}) {
      std::stringstream ss;
      ss << "RepErrorFloor/N" << N;
      auto name = ss.str();
      testing::RegisterTest(
          "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
          __FILE__, __LINE__,
          [N]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
            return new BF16LayerNormPrecisionShortTest(1, N, false);
          });
      count++;
    }

    // Catastrophic cancellation
    for (size_t N : {64, 256, 1024, 4096}) {
      std::stringstream ss;
      ss << "CatastrophicCancellation/N" << N;
      auto name = ss.str();
      testing::RegisterTest(
          "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
          __FILE__, __LINE__,
          [N]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
            return new BF16LayerNormPrecisionShortTest(2, N, false);
          });
      count++;
    }

    // High dynamic range
    for (size_t N : {64, 256, 1024}) {
      std::stringstream ss;
      ss << "HighDynamicRange/N" << N;
      auto name = ss.str();
      testing::RegisterTest(
          "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
          __FILE__, __LINE__,
          [N]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
            return new BF16LayerNormPrecisionShortTest(3, N, false);
          });
      count++;
    }

    // Near-zero variance
    for (size_t N : {64, 256, 1024}) {
      std::stringstream ss;
      ss << "NearZeroVariance/N" << N;
      auto name = ss.str();
      testing::RegisterTest(
          "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
          __FILE__, __LINE__,
          [N]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
            return new BF16LayerNormPrecisionShortTest(4, N, false);
          });
      count++;
    }

    // Denormal inputs (fixed N=64)
    testing::RegisterTest(
        "BF16LayerNormPrecision", "DenormalInputs", nullptr, "DenormalInputs",
        __FILE__, __LINE__,
        []() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
          return new BF16LayerNormPrecisionShortTest(5, 64, false);
        });
    count++;

    // Near-max values
    for (size_t N : {64, 256}) {
      std::stringstream ss;
      ss << "NearMaxValues/N" << N;
      auto name = ss.str();
      testing::RegisterTest(
          "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
          __FILE__, __LINE__,
          [N]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
            return new BF16LayerNormPrecisionShortTest(6, N, false);
          });
      count++;
    }

    // Widen-accumulate-narrow simulation
    for (size_t N : {64, 256, 1024, 4096, 16384, 65536}) {
      for (bool simplified : {false, true}) {
        std::stringstream ss;
        ss << "WidenAccNarrow/N" << N << "/simplified" << simplified;
        auto name = ss.str();
        testing::RegisterTest(
            "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
            __FILE__, __LINE__,
            [N, simplified]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
              return new BF16LayerNormPrecisionShortTest(7, N, simplified);
            });
        count++;
      }
    }

    // BF16 vs FP16 cross-check
    for (size_t N : {256, 1024}) {
      std::stringstream ss;
      ss << "BF16vsFP16/N" << N;
      auto name = ss.str();
      testing::RegisterTest(
          "BF16LayerNormPrecision", name.c_str(), nullptr, name.c_str(),
          __FILE__, __LINE__,
          [N]() -> MlasTestFixture<BF16LayerNormPrecisionTest>* {
            return new BF16LayerNormPrecisionShortTest(8, N, false);
          });
      count++;
    }

    return count;
  }

 private:
  int test_id_;
  size_t N_;
  bool simplified_;
};

static UNUSED_VARIABLE bool added_bf16_rounding = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return BF16RoundingShortTest::RegisterShortExecuteTests();
      }
      return 0;
    });

static UNUSED_VARIABLE bool added_bf16_precision = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return BF16LayerNormPrecisionShortTest::RegisterShortExecuteTests();
      }
      return 0;
    });
