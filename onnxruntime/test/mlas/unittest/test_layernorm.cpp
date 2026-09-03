/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_layernorm.cpp

Abstract:

    Tests for MLAS LayerNorm/RMSNorm (MlasLayerNormF32).

--*/

#include "test_util.h"
#include "mlas.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
// This test's own ScalarLayerNorm oracle (below) intentionally uses the
// single-pass "E[x^2] - mean^2" formula, which is a *different* (but
// equally valid) floating-point evaluation order than a centered two-pass
// "mean first, then sum-of-centered-squares" formula that a registered
// kernel is free to use instead (e.g. the Apple Accelerate/vDSP kernel in
// onnxruntime/core/mlas/lib/layernorm.cpp). The two orders only disagree by
// a handful of ULPs for most inputs, but the uncentered formula suffers
// catastrophic cancellation at small norm_size for this test's fixed input
// pattern (observed: norm_size=7 diverges from a centered kernel by ~2e-3
// absolute in 1/std_dev, versus ~4e-5 relative) -- an artifact of which
// formula the *reference* happens to use, not a kernel defect. A relative
// tolerance (with a small absolute floor for near-zero reference values)
// accepts either valid evaluation order while still catching real bugs:
// a wrong scale factor, dropped bias, or incorrect epsilon placement
// produces mismatches many orders of magnitude larger than the floating-
// point evaluation-order noise this tolerates.
constexpr float kRelTolerance = 1e-3f;
constexpr float kAbsToleranceFloor = 1e-4f;

::testing::AssertionResult NearRelative(const char* actual_expr, const char* expected_expr,
                                        const char* /*rel_expr*/, const char* /*floor_expr*/,
                                        float actual, float expected, float rel_tol, float abs_floor) {
  const float tol = std::max(abs_floor, rel_tol * std::abs(expected));
  const float diff = std::abs(actual - expected);
  if (diff <= tol) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
         << actual_expr << " (" << actual << ") and " << expected_expr << " (" << expected
         << ") are not near enough: diff=" << diff << " > tol=" << tol
         << " (rel_tol=" << rel_tol << ", abs_floor=" << abs_floor << ")";
}
}  // namespace

#define ASSERT_NEAR_REL(actual, expected) \
  ASSERT_PRED_FORMAT4(NearRelative, actual, expected, kRelTolerance, kAbsToleranceFloor)

class MlasLayerNormTest : public MlasTestBase {
 private:
  void ScalarLayerNorm(
      const float* input,
      const float* scale,
      const float* bias,
      float* output,
      float* mean_out,
      float* inv_std_out,
      size_t norm_size,
      float epsilon,
      bool simplified) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < norm_size; i++) {
      sum += input[i];
      sum_sq += input[i] * input[i];
    }
    float mean = sum / static_cast<float>(norm_size);
    float denom;
    if (simplified) {
      denom = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
    } else {
      denom = std::sqrt(sum_sq / static_cast<float>(norm_size) - mean * mean + epsilon);
    }
    float inv_denom = 1.0f / denom;

    for (size_t i = 0; i < norm_size; i++) {
      if (simplified) {
        output[i] = input[i] * inv_denom * scale[i];
      } else if (bias == nullptr) {
        output[i] = (input[i] - mean) * inv_denom * scale[i];
      } else {
        output[i] = (input[i] - mean) * inv_denom * scale[i] + bias[i];
      }
    }
    if (mean_out) *mean_out = mean;
    if (inv_std_out) *inv_std_out = inv_denom;
  }

 public:
  void Test(size_t norm_size, bool simplified, bool with_bias) {
    std::vector<float> input(norm_size);
    std::vector<float> scale(norm_size);
    std::vector<float> bias(norm_size);
    std::vector<float> output_ref(norm_size);
    std::vector<float> output_mlas(norm_size);
    float mean_ref = 0, mean_mlas = 0;
    float inv_std_ref = 0, inv_std_mlas = 0;

    for (size_t i = 0; i < norm_size; i++) {
      input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
      scale[i] = 1.0f + (static_cast<float>(i % 31) - 15.0f) * 0.001f;
      bias[i] = (static_cast<float>(i % 17) - 8.0f) * 0.005f;
    }

    const float* bias_ptr = (with_bias && !simplified) ? bias.data() : nullptr;

    ScalarLayerNorm(input.data(), scale.data(), bias_ptr,
                    output_ref.data(), &mean_ref, &inv_std_ref,
                    norm_size, 1e-5f, simplified);

    bool used = MlasLayerNormF32(input.data(), scale.data(), bias_ptr,
                                 output_mlas.data(), &mean_mlas, &inv_std_mlas,
                                 norm_size, 1e-5f, simplified);

    if (!used) {
      // No optimized kernel available, skip comparison
      return;
    }

    for (size_t i = 0; i < norm_size; i++) {
      ASSERT_NEAR_REL(output_mlas[i], output_ref[i])
          << "output mismatch at [" << i << "], norm_size=" << norm_size
          << " simplified=" << simplified << " bias=" << with_bias;
    }
    ASSERT_NEAR_REL(mean_mlas, mean_ref) << "mean mismatch";
    ASSERT_NEAR_REL(inv_std_mlas, inv_std_ref) << "inv_std_dev mismatch";
  }
};

class LayerNormShortExecuteTest : public MlasTestFixture<MlasLayerNormTest> {
 public:
  LayerNormShortExecuteTest(size_t norm_size, bool simplified, bool with_bias)
      : norm_size_(norm_size), simplified_(simplified), with_bias_(with_bias) {}

  void TestBody() override {
    MlasTestFixture<MlasLayerNormTest>::mlas_tester->Test(norm_size_, simplified_, with_bias_);
  }

  static size_t RegisterSingleTest(size_t norm_size, bool simplified, bool with_bias) {
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

  static size_t RegisterShortExecuteTests() {
    size_t count = 0;
    for (size_t n : {1, 7, 32, 63, 64, 127, 128, 256, 1024}) {
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

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return LayerNormShortExecuteTest::RegisterShortExecuteTests();
      }
      return 0;
    });
