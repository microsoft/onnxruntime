// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the Apple Accelerate (vDSP) LayerNorm/RMSNorm kernel added
// behind the onnxruntime_USE_APPLE_ACCELERATE CMake option. This file
// compiles to nothing unless the build is targeting macOS arm64 with that
// option enabled -- matching the #if-gating pattern already used elsewhere
// for Apple/ARM64-only MLAS unit tests. On every other configuration
// (including the default build, where the option is OFF) this translation
// unit is an empty no-op.
//
// Unlike Tanh (an elementwise function where every output element depends
// only on the corresponding input element), LayerNorm/RMSNorm is a whole-row
// reduction: every output element depends on the mean and/or sum-of-squares
// of the ENTIRE row. That has two test-design consequences used throughout
// this file:
//
//   1. A fp64-accumulated scalar reference (ReferenceLayerNorm below) that
//      deliberately uses the SAME centered two-pass formula as the kernel
//      under test (mean first, then sum of centered squares) but computed
//      independently in double precision rather than by calling into any
//      MLAS/vDSP code. This is a meaningfully independent oracle even though
//      the algorithm shape matches: fp64 accumulation over these test sizes
//      cannot experience the fp32 catastrophic-cancellation failure mode
//      that motivated choosing the centered formula in the first place (see
//      the kernel's own doc comment in layernorm.cpp), so agreement between
//      the fp64 reference and the fp32 kernel is a genuine correctness
//      check, not merely "did I retype the same formula".
//
//   2. Because it is a reduction, IEEE special values (a single NaN or Inf
//      anywhere in the row) contaminate the ENTIRE output row through the
//      shared mean/variance, rather than only the one element that was
//      special. Tests below assert that behavior directly against the fp64
//      reference (which exhibits the identical contamination for the same
//      mathematical reason: Inf-Inf=NaN, NaN propagates through +), instead
//      of hand-picking an expected value the way the Tanh tests could for
//      an elementwise function.

#include "gtest/gtest.h"

#include "core/mlas/lib/mlasi.h"

#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

// Poison value used to pre-fill output buffers before calling a kernel
// under test, instead of the natural-looking 0.0f: a kernel that silently
// fails to write an element must fail loudly rather than pass by
// coincidence.
//
// LayerNorm/RMSNorm's real output range is not bounded to [-1, 1] the way
// Tanh's is, so a much larger finite sentinel is required to guarantee it
// can never coincide with a genuinely correct answer. All test inputs in
// this file are constructed with magnitude well under 1e6 and Scale/Bias
// magnitude under 10, so even an adversarial near-zero denominator (as
// small as sqrt(Epsilon) ~ 3e-3 for the smallest Epsilon used here) cannot
// produce a genuine output anywhere near 1e30.
constexpr float kPoisonValue = 1.0e30f;

// fp64-accumulated scalar reference, using the SAME centered two-pass
// formula as the kernel under test (see file-level comment for why this is
// still a meaningfully independent oracle).
void ReferenceLayerNorm(
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
  for (size_t i = 0; i < norm_size; i++) {
    sum += static_cast<double>(input[i]);
  }
  double mean = sum / static_cast<double>(norm_size);

  double denom;
  if (simplified) {
    double mean_sq = 0.0;
    for (size_t i = 0; i < norm_size; i++) {
      double x = static_cast<double>(input[i]);
      mean_sq += x * x;
    }
    mean_sq /= static_cast<double>(norm_size);
    denom = std::sqrt(mean_sq + static_cast<double>(epsilon));
  } else {
    double sum_sq = 0.0;
    for (size_t i = 0; i < norm_size; i++) {
      double c = static_cast<double>(input[i]) - mean;
      sum_sq += c * c;
    }
    denom = std::sqrt(sum_sq / static_cast<double>(norm_size) + static_cast<double>(epsilon));
  }
  double inv_denom = 1.0 / denom;

  for (size_t i = 0; i < norm_size; i++) {
    double centered = simplified ? static_cast<double>(input[i]) : (static_cast<double>(input[i]) - mean);
    double y = centered * inv_denom * static_cast<double>(scale[i]);
    if (bias != nullptr) {
      y += static_cast<double>(bias[i]);
    }
    output[i] = static_cast<float>(y);
  }
  if (mean_out != nullptr) {
    *mean_out = static_cast<float>(mean);
  }
  if (inv_std_out != nullptr) {
    *inv_std_out = static_cast<float>(inv_denom);
  }
}

// Relative tolerance matching upstream's CloseEnough (rel_tol=0.005), with a
// 1e-4 absolute floor. 1/sqrt(var+eps) amplifies small variance differences
// between fp32 vDSP accumulation and the fp64 reference, particularly for
// small NormSize where variance is near zero.
bool NearEnough(float got, float ref) {
  if (std::isnan(got) || std::isnan(ref)) {
    return std::isnan(got) && std::isnan(ref);
  }
  if (std::isinf(got) || std::isinf(ref)) {
    // fabs(inf - inf) is NaN, which would otherwise make this function
    // incorrectly reject two equal infinities of the same sign (not
    // currently reachable by any caller in this file, since Inf-producing
    // cases are asserted directly rather than through this helper, but
    // guarded here defensively).
    return got == ref;
  }
  float diff = std::fabs(got - ref);
  if (diff <= 1e-4f) {
    return true;
  }
  float top = std::max(std::fabs(got), std::fabs(ref));
  return (top > 1e-6f) && (diff / top < 0.005f);
}

// Deterministic, non-adversarial fill exercising positive, negative, and
// near-zero values, matching the style already established by the
// pre-existing generic test_layernorm.cpp.
void FillDeterministic(std::vector<float>& input, std::vector<float>& scale, std::vector<float>& bias) {
  size_t n = input.size();
  scale.resize(n);
  bias.resize(n);
  for (size_t i = 0; i < n; i++) {
    input[i] = (static_cast<float>(i % 127) - 63.0f) * 0.01f;
    scale[i] = 1.0f + (static_cast<float>(i % 31) - 15.0f) * 0.001f;
    bias[i] = (static_cast<float>(i % 17) - 8.0f) * 0.005f;
  }
}

// One test body shared by ForcedReachability and LargeNormSizeHeapFallback:
// calls MlasLayerNormKernelAppleAccelerate directly (bypassing the
// GetMlasPlatform() dispatch table) and compares against the independent
// fp64 reference.
void RunDirectKernelParityCase(size_t norm_size, bool simplified, bool with_bias, float epsilon) {
  std::vector<float> input(norm_size), scale, bias;
  FillDeterministic(input, scale, bias);
  const float* bias_ptr = (with_bias && !simplified) ? bias.data() : nullptr;

  std::vector<float> output_ref(norm_size), output_kernel(norm_size, kPoisonValue);
  float mean_ref = 0.0f, mean_kernel = 0.0f;
  float inv_std_ref = 0.0f, inv_std_kernel = 0.0f;

  ReferenceLayerNorm(input.data(), scale.data(), bias_ptr, output_ref.data(), &mean_ref, &inv_std_ref, norm_size,
                     epsilon, simplified);
  MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias_ptr, output_kernel.data(), &mean_kernel,
                                     &inv_std_kernel, norm_size, epsilon, simplified);

  for (size_t i = 0; i < norm_size; i++) {
    EXPECT_TRUE(NearEnough(output_kernel[i], output_ref[i]))
        << "output mismatch at [" << i << "], norm_size=" << norm_size << " simplified=" << simplified
        << " bias=" << with_bias << " got=" << output_kernel[i] << " ref=" << output_ref[i];
  }
  EXPECT_TRUE(NearEnough(mean_kernel, mean_ref)) << "mean mismatch, norm_size=" << norm_size;
  EXPECT_TRUE(NearEnough(inv_std_kernel, inv_std_ref)) << "inv_std_dev mismatch, norm_size=" << norm_size;
}

}  // namespace

// Forced-reachability test: calls MlasLayerNormKernelAppleAccelerate
// directly, proving the vDSP-backed symbol itself links and is callable
// whenever this translation unit is built, independent of whether the
// GetMlasPlatform() registration in platform.cpp is correct. The companion
// PublicDispatchMatchesDirectKernelCall test below covers that wiring.
// Covers both Simplified (RMSNorm) and full LayerNorm, with and without
// bias, across a range of NormSize including 1 (degenerate row), values
// that are and are not multiples of 4 (vDSP is not a fixed-width-vector
// API, but this matches the sizes exercised by the pre-existing generic
// LayerNorm test suite), a representative sample of real transformer
// hidden dims (768 BERT/GPT-2, 3072 Phi-3/Gemma, 4096 Llama), and the exact
// kApplePerRowStackScratch boundary (8192, see layernorm.cpp): NormSize ==
// 8192 takes the on-stack scratch path (the condition is strictly
// greater-than), so it belongs here rather than in
// LargeNormSizeHeapFallback below, which covers NormSize > 8192.
TEST(LayerNormAppleAccelerate, ForcedReachability) {
  for (size_t norm_size : {size_t{1}, size_t{7}, size_t{8}, size_t{63}, size_t{64}, size_t{127}, size_t{128},
                           size_t{768}, size_t{1024}, size_t{3072}, size_t{4096}, size_t{8192}}) {
    for (bool simplified : {true, false}) {
      for (bool with_bias : {true, false}) {
        RunDirectKernelParityCase(norm_size, simplified, with_bias, 1e-5f);
      }
    }
  }
}

// Exercises the heap-allocated scratch buffer fallback path: NormSize
// exceeding kApplePerRowStackScratch (8192) in layernorm.cpp. Every case
// above this test stays within the on-stack scratch buffer; this is the
// only test that forces the std::unique_ptr<float[]> heap path, so it is
// the sole regression coverage for that branch.
TEST(LayerNormAppleAccelerate, LargeNormSizeHeapFallback) {
  for (size_t norm_size : {size_t{8193}, size_t{16384}}) {
    for (bool simplified : {true, false}) {
      RunDirectKernelParityCase(norm_size, simplified, /*with_bias=*/!simplified, 1e-5f);
    }
  }
}

// End-to-end dispatch test: proves the public MlasLayerNormF32 entry point
// declines rows below the measured 64-element crossover so the caller uses
// its scalar fallback, and resolves to the Apple kernel at and above the
// threshold. A compile-time #if cannot be introspected at runtime, so
// selected sizes are compared with a direct kernel call.
TEST(LayerNormAppleAccelerate, PublicDispatchHonorsCrossoverAndMatchesDirectKernelCall) {
  ASSERT_NE(GetMlasPlatform().LayerNormF32Kernel, nullptr)
      << "This translation unit is only built when MLAS_USE_APPLE_ACCELERATE is enabled on macOS arm64; "
      << "the kernel must be registered in platform.cpp's MLAS_TARGET_ARM64 block whenever it is built.";

  constexpr size_t kMinimumDispatchSize = 64;
  for (size_t norm_size : {size_t{1}, size_t{63}, size_t{64}, size_t{127}, size_t{768}, size_t{4096}}) {
    for (bool simplified : {true, false}) {
      std::vector<float> input(norm_size), scale, bias;
      FillDeterministic(input, scale, bias);
      const float* bias_ptr = simplified ? nullptr : bias.data();

      std::vector<float> dispatch_output(norm_size, kPoisonValue);
      std::vector<float> direct_output(norm_size, kPoisonValue);
      float dispatch_mean = kPoisonValue, direct_mean = 0.0f;
      float dispatch_inv_std = kPoisonValue, direct_inv_std = 0.0f;

      bool used = MlasLayerNormF32(input.data(), scale.data(), bias_ptr, dispatch_output.data(), &dispatch_mean,
                                   &dispatch_inv_std, norm_size, 1e-5f, simplified);
      if (norm_size < kMinimumDispatchSize) {
        EXPECT_FALSE(used) << "vDSP should not be selected below its measured crossover";
        EXPECT_EQ(dispatch_mean, kPoisonValue);
        EXPECT_EQ(dispatch_inv_std, kPoisonValue);
        for (float value : dispatch_output) {
          EXPECT_EQ(value, kPoisonValue);
        }
        continue;
      }

      ASSERT_TRUE(used) << "REACHABILITY FAILURE: MlasLayerNormF32 returned false even though a kernel is "
                        << "registered (norm_size=" << norm_size << ").";

      MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias_ptr, direct_output.data(), &direct_mean,
                                         &direct_inv_std, norm_size, 1e-5f, simplified);

      for (size_t i = 0; i < norm_size; i++) {
        EXPECT_EQ(dispatch_output[i], direct_output[i])
            << "MlasLayerNormF32 dispatch diverged from MlasLayerNormKernelAppleAccelerate at i=" << i
            << ", norm_size=" << norm_size << "; check the registration in platform.cpp.";
      }
      EXPECT_EQ(dispatch_mean, direct_mean);
      EXPECT_EQ(dispatch_inv_std, direct_inv_std);
    }
  }
}

// Confirms the in-place aliasing contract (Input == Output) that real ORT
// LayerNorm/SkipLayerNorm call sites rely on (see the kernel's own doc
// comment in layernorm.cpp for why this is safe by construction: Output is
// only ever written by the final vDSP_vma/vDSP_vmul call, whose operands
// are never Input directly).
TEST(LayerNormAppleAccelerate, InPlaceAliasing) {
  for (size_t norm_size : {size_t{1}, size_t{127}, size_t{768}, size_t{4096}}) {
    for (bool simplified : {true, false}) {
      for (bool with_bias : {true, false}) {
        std::vector<float> original_input(norm_size), scale, bias;
        FillDeterministic(original_input, scale, bias);
        const float* bias_ptr = (with_bias && !simplified) ? bias.data() : nullptr;

        std::vector<float> aliased_buffer = original_input;
        MlasLayerNormKernelAppleAccelerate(aliased_buffer.data(), scale.data(), bias_ptr, aliased_buffer.data(),
                                           nullptr, nullptr, norm_size, 1e-5f, simplified);

        std::vector<float> separate_output(norm_size, kPoisonValue);
        MlasLayerNormKernelAppleAccelerate(original_input.data(), scale.data(), bias_ptr, separate_output.data(),
                                           nullptr, nullptr, norm_size, 1e-5f, simplified);

        for (size_t i = 0; i < norm_size; i++) {
          EXPECT_EQ(aliased_buffer[i], separate_output[i])
              << "in-place (Input==Output) result diverged from separate-buffer result at i=" << i
              << ", norm_size=" << norm_size << " simplified=" << simplified << " bias=" << with_bias;
        }
      }
    }
  }
}

// No-write detection: pre-fills Output with a poison value before calling
// the kernel and asserts every element was overwritten. See the file-level
// comment on kPoisonValue for why 1e30f can never coincide with a genuine
// output for these test inputs.
TEST(LayerNormAppleAccelerate, NoWriteDetectionPoisonBuffer) {
  for (size_t norm_size : {size_t{1}, size_t{63}, size_t{768}, size_t{4096}, size_t{8193}}) {
    for (bool simplified : {true, false}) {
      std::vector<float> input(norm_size), scale, bias;
      FillDeterministic(input, scale, bias);
      const float* bias_ptr = simplified ? nullptr : bias.data();

      std::vector<float> output(norm_size, kPoisonValue);
      MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias_ptr, output.data(), nullptr, nullptr,
                                         norm_size, 1e-5f, simplified);

      for (size_t i = 0; i < norm_size; i++) {
        EXPECT_NE(output[i], kPoisonValue)
            << "element " << i << " was never written (still holds the poison sentinel), norm_size=" << norm_size;
      }
    }
  }
}

// Zero-variance edge case: every element in the row is identical, so the
// true variance (full LayerNorm) is exactly zero. This exercises the
// epsilon-only denominator path (denom = sqrt(0 + epsilon)) and confirms no
// NaN/Inf results from it. For RMSNorm (simplified), a constant nonzero row
// has nonzero mean-of-squares, so this instead exercises the ordinary path
// with a degenerate (zero-spread) input.
TEST(LayerNormAppleAccelerate, ZeroVarianceConstantRow) {
  for (size_t norm_size : {size_t{1}, size_t{127}, size_t{768}}) {
    for (bool simplified : {true, false}) {
      std::vector<float> input(norm_size, 3.5f);
      std::vector<float> scale(norm_size, 1.0f);
      std::vector<float> bias(norm_size, 0.25f);
      const float* bias_ptr = simplified ? nullptr : bias.data();

      std::vector<float> output_ref(norm_size), output_kernel(norm_size, kPoisonValue);
      ReferenceLayerNorm(input.data(), scale.data(), bias_ptr, output_ref.data(), nullptr, nullptr, norm_size, 1e-5f,
                         simplified);
      MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias_ptr, output_kernel.data(), nullptr,
                                         nullptr, norm_size, 1e-5f, simplified);

      for (size_t i = 0; i < norm_size; i++) {
        ASSERT_FALSE(std::isnan(output_kernel[i])) << "unexpected NaN at i=" << i << ", norm_size=" << norm_size;
        ASSERT_FALSE(std::isinf(output_kernel[i])) << "unexpected Inf at i=" << i << ", norm_size=" << norm_size;
        EXPECT_TRUE(NearEnough(output_kernel[i], output_ref[i]))
            << "output mismatch at [" << i << "], norm_size=" << norm_size << " got=" << output_kernel[i]
            << " ref=" << output_ref[i];
      }
    }
  }
}

// A single NaN anywhere in the row must contaminate the entire output row
// through the shared mean/sum-of-squares reduction -- this is an inherent
// property of the mathematical formula (NaN propagates through +), not a
// vDSP-specific behavior, so it is verified against the fp64 reference
// (which exhibits identical contamination) rather than a hand-picked
// expectation.
TEST(LayerNormAppleAccelerate, NanPropagation) {
  for (size_t norm_size : {size_t{2}, size_t{127}, size_t{768}}) {
    for (bool simplified : {true, false}) {
      std::vector<float> input(norm_size), scale, bias;
      FillDeterministic(input, scale, bias);
      input[norm_size / 2] = std::numeric_limits<float>::quiet_NaN();
      const float* bias_ptr = simplified ? nullptr : bias.data();

      std::vector<float> output(norm_size, kPoisonValue);
      MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias_ptr, output.data(), nullptr, nullptr,
                                         norm_size, 1e-5f, simplified);

      for (size_t i = 0; i < norm_size; i++) {
        EXPECT_TRUE(std::isnan(output[i]))
            << "expected NaN contamination of the entire row at i=" << i << ", norm_size=" << norm_size
            << " simplified=" << simplified << ", got " << output[i];
      }
    }
  }
}

// A single +Inf anywhere in the row propagates differently depending on
// whether the mean is subtracted:
//
// - Full LayerNorm (simplified=false): mean becomes Inf (a finite sum plus
//   Inf is still Inf), so *every* element's centering computes
//   (finite - Inf) = -Inf for the non-Inf elements and (Inf - Inf) = NaN for
//   the Inf element itself. The sum-of-centered-squares reduction therefore
//   includes a NaN term, which poisons the shared variance/inv-std for the
//   entire row: every output element becomes NaN.
// - RMSNorm (simplified=true): there is no mean subtraction, so
//   mean-of-squares is simply Inf (again, finite-sum-plus-Inf is Inf), and
//   1/sqrt(Inf + eps) is exactly 0.0. Each *non*-Inf element then computes
//   finite * 0.0 = 0.0 (not NaN -- there is no Inf-Inf cancellation on that
//   path), while the Inf element itself computes Inf * 0.0 = NaN. So only
//   the single position that was set to +Inf becomes NaN; every other
//   position is exactly 0.0.
//
// This was caught by real Apple Silicon CI (this test originally asserted
// whole-row NaN contamination unconditionally, which is only correct for
// the non-simplified path) -- see PR discussion for the CI run this was
// caught in. Verified independently in Python/IEEE-754 semantics before
// fixing: 1.0/sqrt(inf) == 0.0, finite*0.0 == 0.0, inf*0.0 == nan.
TEST(LayerNormAppleAccelerate, InfPropagation) {
  for (size_t norm_size : {size_t{2}, size_t{127}, size_t{768}}) {
    for (bool simplified : {true, false}) {
      const size_t inf_index = norm_size / 2;
      std::vector<float> input(norm_size), scale, bias;
      FillDeterministic(input, scale, bias);
      input[inf_index] = std::numeric_limits<float>::infinity();
      const float* bias_ptr = simplified ? nullptr : bias.data();

      std::vector<float> output(norm_size, kPoisonValue);
      MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias_ptr, output.data(), nullptr, nullptr,
                                         norm_size, 1e-5f, simplified);

      for (size_t i = 0; i < norm_size; i++) {
        if (!simplified || i == inf_index) {
          EXPECT_TRUE(std::isnan(output[i]))
              << "expected NaN at i=" << i << ", norm_size=" << norm_size << " simplified=" << simplified
              << ", got " << output[i];
        } else {
          EXPECT_EQ(output[i], 0.0f) << "expected exact 0.0 (finite * 1/sqrt(Inf) == finite * 0.0) at i=" << i
                                     << ", norm_size=" << norm_size << ", got " << output[i];
        }
      }
    }
  }
}

// Denormal/near-zero input row: every element is a subnormal float. This
// probes the same divergent hardware behaviors documented in the sibling
// Tanh test (IEEE-correctly-rounded vs. flush-to-zero), but for a
// reduction rather than an elementwise function: unlike Tanh, where FTZ can
// be assessed per-element, here an FTZ-on-input behavior would make the
// entire row read as all-zero, which is a well-defined, finite, non-NaN
// outcome (zero mean, zero variance, denom=sqrt(epsilon)). This test
// therefore only requires a finite, non-NaN result -- it does not assume
// either behavior, matching the sibling test's "don't invent unverified
// hardware behavior" rationale.
TEST(LayerNormAppleAccelerate, DenormalInputs) {
  for (size_t norm_size : {size_t{4}, size_t{127}}) {
    for (bool simplified : {true, false}) {
      std::vector<float> input(norm_size);
      for (size_t i = 0; i < norm_size; i++) {
        input[i] = (i % 2 == 0) ? std::numeric_limits<float>::denorm_min() : -std::numeric_limits<float>::denorm_min();
      }
      std::vector<float> scale(norm_size, 1.0f);
      std::vector<float> bias(norm_size, 0.0f);
      const float* bias_ptr = simplified ? nullptr : bias.data();

      std::vector<float> output(norm_size, kPoisonValue);
      MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias_ptr, output.data(), nullptr, nullptr,
                                         norm_size, 1e-5f, simplified);

      for (size_t i = 0; i < norm_size; i++) {
        EXPECT_FALSE(std::isnan(output[i])) << "input=denormal, got NaN output at i=" << i;
        EXPECT_FALSE(std::isinf(output[i])) << "input=denormal, got Inf output at i=" << i;
      }
    }
  }
}

// NormSize == 0 is an explicit early-return edge case in the kernel (see
// layernorm.cpp): it must not read/write Input/Output at all, and must
// write 0.0f to MeanOut/InvStdDevOut when they are non-null, rather than
// leaving them uninitialized or dividing by zero.
TEST(LayerNormAppleAccelerate, ZeroNormSizeNoWrite) {
  float mean_out = kPoisonValue;
  float inv_std_out = kPoisonValue;
  // Input/Output/Scale/Bias are deliberately nullptr: NormSize == 0 means
  // the kernel must never dereference them.
  MlasLayerNormKernelAppleAccelerate(nullptr, nullptr, nullptr, nullptr, &mean_out, &inv_std_out, 0, 1e-5f, false);
  EXPECT_EQ(mean_out, 0.0f);
  EXPECT_EQ(inv_std_out, 0.0f);

  // Simplified (RMSNorm) path must behave identically for NormSize == 0.
  mean_out = kPoisonValue;
  inv_std_out = kPoisonValue;
  MlasLayerNormKernelAppleAccelerate(nullptr, nullptr, nullptr, nullptr, &mean_out, &inv_std_out, 0, 1e-5f, true);
  EXPECT_EQ(mean_out, 0.0f);
  EXPECT_EQ(inv_std_out, 0.0f);
}

// MeanOut/InvStdDevOut are optional (nullptr is a valid, commonly-used
// argument -- e.g. every RMSNorm caller passes nullptr for MeanOut, since
// the ONNX SimplifiedLayerNormalization contract has no Mean output). This
// confirms passing nullptr for both never crashes and the Output values are
// still correct.
TEST(LayerNormAppleAccelerate, NullMeanAndInvStdDevOutputs) {
  constexpr size_t kNormSize = 256;
  std::vector<float> input(kNormSize), scale, bias;
  FillDeterministic(input, scale, bias);

  std::vector<float> output_ref(kNormSize), output_kernel(kNormSize, kPoisonValue);
  ReferenceLayerNorm(input.data(), scale.data(), bias.data(), output_ref.data(), nullptr, nullptr, kNormSize, 1e-5f,
                     false);
  MlasLayerNormKernelAppleAccelerate(input.data(), scale.data(), bias.data(), output_kernel.data(), nullptr, nullptr,
                                     kNormSize, 1e-5f, false);

  for (size_t i = 0; i < kNormSize; i++) {
    EXPECT_TRUE(NearEnough(output_kernel[i], output_ref[i])) << "output mismatch at [" << i << "]";
  }
}

#endif  // MLAS_USE_APPLE_ACCELERATE && __APPLE__ && MLAS_TARGET_ARM64
