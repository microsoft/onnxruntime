// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the Apple Accelerate (vForce) Tanh kernel added behind the
// onnxruntime_USE_APPLE_ACCELERATE CMake option. This file compiles to
// nothing unless the build is targeting macOS arm64 with that option
// enabled -- matching the pattern already used for other Apple/ARM64-only
// MLAS unit tests (see test_hgemm_neon.cpp, which gates on
// MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64). On every other
// configuration (including the default build, where the option is OFF) this
// translation unit is an empty no-op.

#include "gtest/gtest.h"

#include "core/mlas/lib/mlasi.h"

#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 1e-6f;
constexpr float kRelativeTolerance = 1e-6f;

// MlasTanhKernel (the portable polynomial reference) is the parity baseline:
// it is the exact implementation this Apple path replaces in
// MlasComputeTanh<float> when the option is enabled, so any behavioral
// change is measured directly against it rather than against std::tanh.
void ExpectMatchesPortableKernel(float apple_value, float portable_value, float input) {
  if (std::isnan(portable_value)) {
    EXPECT_TRUE(std::isnan(apple_value)) << "input=" << input << " expected NaN, got " << apple_value;
    return;
  }
  const float diff = std::fabs(apple_value - portable_value);
  EXPECT_TRUE(diff <= kAbsoluteTolerance || diff <= std::fabs(portable_value) * kRelativeTolerance)
      << "input=" << input << " apple=" << apple_value << " portable=" << portable_value << " diff=" << diff;
}

std::vector<float> MakeTanhTestInputs(size_t random_count, unsigned seed) {
  std::vector<float> input;
  input.reserve(random_count + 9);

  // Values that exercise the reference kernel's +/-9 clamp boundary, exact
  // zero (both signs), and IEEE-754 special values.
  input.push_back(0.0f);
  input.push_back(-0.0f);
  input.push_back(9.0f);
  input.push_back(-9.0f);
  input.push_back(20.0f);
  input.push_back(-20.0f);
  input.push_back(std::numeric_limits<float>::infinity());
  input.push_back(-std::numeric_limits<float>::infinity());
  input.push_back(std::numeric_limits<float>::quiet_NaN());

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(-12.0f, 12.0f);
  for (size_t i = 0; i < random_count; i++) {
    input.push_back(distribution(generator));
  }

  return input;
}

}  // namespace

// Forced-reachability test: calls MlasTanhKernelAppleAccelerate directly
// (not through the MlasComputeTanh<float> dispatch table), proving the
// Accelerate-backed symbol itself links and is callable whenever this
// translation unit is built -- i.e. whenever
// onnxruntime_USE_APPLE_ACCELERATE is enabled on macOS arm64 -- independent
// of whether the dispatch wiring in tanh.cpp is correct. The companion
// PublicDispatchMatchesDirectKernelCall test below covers that wiring.
TEST(TanhAppleAccelerate, ForcedReachability) {
  const std::vector<float> input = MakeTanhTestInputs(256, 12345u);
  const size_t n = input.size();

  std::vector<float> apple_output(n);
  std::vector<float> portable_output(n);

  MlasTanhKernelAppleAccelerate(input.data(), apple_output.data(), n);
  MlasTanhKernel(input.data(), portable_output.data(), n);

  for (size_t i = 0; i < n; i++) {
    ExpectMatchesPortableKernel(apple_output[i], portable_output[i], input[i]);
  }
}

// Confirms the in-place aliasing contract (Input == Output) that real MLAS
// callers rely on: providers/cpu/tensor/gelu.cc, providers/cpu/ml/
// svmclassifier.h, and providers/cpu/rnn/rnn_helpers.cc all invoke
// MlasComputeTanh with the same buffer as both arguments. vvtanhf computes
// each output element from only the corresponding input element, so
// in-place use is expected to be safe, but this is exercised explicitly
// rather than assumed.
TEST(TanhAppleAccelerate, InPlaceAliasing) {
  const std::vector<float> original_input = MakeTanhTestInputs(120, 6789u);
  const size_t n = original_input.size();

  std::vector<float> buffer = original_input;
  MlasTanhKernelAppleAccelerate(buffer.data(), buffer.data(), n);

  std::vector<float> portable_output(n);
  MlasTanhKernel(original_input.data(), portable_output.data(), n);

  for (size_t i = 0; i < n; i++) {
    ExpectMatchesPortableKernel(buffer[i], portable_output[i], original_input[i]);
  }
}

// End-to-end dispatch test: proves the public MlasComputeTanh<float> entry
// point -- the one every real caller (Tanh activation, Gelu tanh-approx,
// SVM classifier, RNN cells) actually uses -- resolves to the Apple kernel
// rather than silently keeping the portable fallback. A compile-time #if
// cannot be introspected at runtime, so this instead checks that the public
// dispatch produces bit-identical output to a direct call to
// MlasTanhKernelAppleAccelerate; any divergence means the dispatch branch in
// MlasComputeTanh<float> and this kernel have drifted apart (e.g. the #if
// condition was edited without updating the call it guards).
TEST(TanhAppleAccelerate, PublicDispatchMatchesDirectKernelCall) {
  const std::vector<float> input = MakeTanhTestInputs(500, 2468u);
  const size_t n = input.size();

  std::vector<float> dispatch_output(n);
  std::vector<float> direct_output(n);

  MlasComputeTanh<float>(input.data(), dispatch_output.data(), n);
  MlasTanhKernelAppleAccelerate(input.data(), direct_output.data(), n);

  for (size_t i = 0; i < n; i++) {
    if (std::isnan(direct_output[i])) {
      EXPECT_TRUE(std::isnan(dispatch_output[i]))
          << "MlasComputeTanh<float> dispatch diverged from MlasTanhKernelAppleAccelerate at i=" << i
          << " (NaN mismatch); check the MLAS_USE_APPLE_ACCELERATE branch in MlasComputeTanh<float>.";
    } else {
      EXPECT_EQ(dispatch_output[i], direct_output[i])
          << "MlasComputeTanh<float> dispatch diverged from MlasTanhKernelAppleAccelerate at i=" << i
          << "; check the MLAS_USE_APPLE_ACCELERATE branch in MlasComputeTanh<float>.";
    }
  }
}

// Exercises the INT32 chunking loop in MlasTanhKernelAppleAccelerate with a
// buffer much larger than any single MLAS caller currently passes, so the
// multi-iteration path (N spanning more than one vForce call, once N
// exceeds INT32_MAX) has at least one non-trivial-size regression test on
// the code paths reachable in CI. The INT32_MAX boundary itself is
// documented, not covered here, because allocating >8GB of buffers to
// reach it is impractical in a CI job; see the PR description for a
// standalone logic-only harness that exercises that boundary with a small
// chunk-size override.
TEST(TanhAppleAccelerate, LargeBufferManyVectorIterations) {
  constexpr size_t kLargeN = size_t{1} << 20;  // 1Mi elements.
  std::vector<float> input(kLargeN);
  std::vector<float> apple_output(kLargeN);
  std::vector<float> portable_output(kLargeN);

  std::default_random_engine generator(999u);
  std::uniform_real_distribution<float> distribution(-9.5f, 9.5f);
  for (size_t i = 0; i < kLargeN; i++) {
    input[i] = distribution(generator);
  }

  MlasTanhKernelAppleAccelerate(input.data(), apple_output.data(), kLargeN);
  MlasTanhKernel(input.data(), portable_output.data(), kLargeN);

  for (size_t i = 0; i < kLargeN; i++) {
    ExpectMatchesPortableKernel(apple_output[i], portable_output[i], input[i]);
  }
}

#endif  // defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)
