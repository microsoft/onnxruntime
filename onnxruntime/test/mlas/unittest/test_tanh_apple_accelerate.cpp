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

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 1e-6f;
constexpr float kRelativeTolerance = 1e-6f;

// MlasTanhKernel (the portable polynomial reference) is the parity baseline
// for ORDINARY FINITE inputs: it is the exact implementation this Apple path
// replaces in MlasComputeTanh<float> when the option is enabled, so any
// behavioral change on finite values (including the +/-9 clamp-boundary
// region, which is a kernel-contract detail, not an IEEE special value) is
// measured directly against it.
//
// IEEE/ONNX special values (NaN, +/-Inf, signed zero) are deliberately NOT
// checked here -- see ExpectIeeeTanhSemantics below. Requiring bit-parity
// with the portable kernel for those would only prove "matches whatever this
// other kernel's clamp/min-max sequence happens to produce", which the
// portable kernel's own scalar-tail comment documents as unreliable for NaN
// in its vectorized (N>=4) path. A semantic assertion against the Tanh
// operator's own contract is direct evidence; a portable-kernel-parity
// assertion is not.
void ExpectMatchesPortableKernel(float apple_value, float portable_value, float input) {
  ASSERT_FALSE(std::isnan(input)) << "input=" << input << ": use ExpectIeeeTanhSemantics for special values";
  const float diff = std::fabs(apple_value - portable_value);
  EXPECT_TRUE(diff <= kAbsoluteTolerance || diff <= std::fabs(portable_value) * kRelativeTolerance)
      << "input=" << input << " apple=" << apple_value << " portable=" << portable_value << " diff=" << diff;
}

// Asserts the IEEE-754 / ONNX Tanh operator contract directly on a kernel's
// output for a given special input, independent of what any other kernel
// implementation (e.g. the portable polynomial reference) happens to
// produce for the same input:
//   * NaN in  -> NaN out. Payload/sign are deliberately unasserted: which
//     NaN vvtanhf returns is Apple's own implementation detail, not part of
//     the operator contract (same rationale as the sibling f16-cast kernel's
//     NaN-payload handling).
//   * +Inf in -> exactly +1.0f out. tanh saturates at the limit; MLAS's own
//     [-9, 9] clamp range was chosen so evaluating the reference polynomial
//     at the boundary already rounds to exactly +/-1.0f in float32, so exact
//     equality (not a tolerance) is the correct assertion.
//   * -Inf in -> exactly -1.0f out.
//   * +0.0 in -> +0.0f out, sign bit clear.
//   * -0.0 in -> -0.0f out, sign bit set. tanh is an odd function and sign
//     of zero is preserved by ordinary float arithmetic, so this is exact,
//     not a tolerance-based check.
void ExpectIeeeTanhSemantics(float value, float input) {
  if (std::isnan(input)) {
    EXPECT_TRUE(std::isnan(value)) << "input=NaN, expected NaN output, got " << value;
  } else if (input == std::numeric_limits<float>::infinity()) {
    EXPECT_EQ(value, 1.0f) << "input=+Inf, expected exactly +1.0f, got " << value;
  } else if (input == -std::numeric_limits<float>::infinity()) {
    EXPECT_EQ(value, -1.0f) << "input=-Inf, expected exactly -1.0f, got " << value;
  } else if (input == 0.0f) {
    EXPECT_EQ(value, 0.0f) << "input=" << (std::signbit(input) ? "-0.0" : "+0.0")
                           << ", expected output magnitude 0, got " << value;
    EXPECT_EQ(std::signbit(value), std::signbit(input))
        << "input=" << (std::signbit(input) ? "-0.0" : "+0.0")
        << ", expected sign-of-zero to be preserved, got signbit=" << std::signbit(value);
  } else {
    FAIL() << "ExpectIeeeTanhSemantics called with a non-special input=" << input
           << "; use ExpectMatchesPortableKernel for ordinary finite values.";
  }
}

// Purely finite inputs (no IEEE special values) for the finite-value
// parity-against-portable-baseline tests. Includes the +/-9 clamp-boundary
// region, a finite MLAS-kernel-contract detail (not an IEEE special value),
// since the Apple kernel must match the portable kernel's approximation
// error there.
std::vector<float> MakeFiniteRandomInputs(size_t random_count, unsigned seed) {
  std::vector<float> input;
  input.reserve(random_count + 4);

  input.push_back(9.0f);
  input.push_back(-9.0f);
  input.push_back(20.0f);
  input.push_back(-20.0f);

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(-12.0f, 12.0f);
  for (size_t i = 0; i < random_count; i++) {
    input.push_back(distribution(generator));
  }

  return input;
}

// IEEE-754 / ONNX Tanh special values, asserted via ExpectIeeeTanhSemantics.
std::vector<float> MakeSpecialValueInputs() {
  return {
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::signaling_NaN(),
  };
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
  const std::vector<float> input = MakeFiniteRandomInputs(256, 12345u);
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
  const std::vector<float> original_input = MakeFiniteRandomInputs(120, 6789u);
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
  std::vector<float> input = MakeSpecialValueInputs();
  const std::vector<float> finite_random = MakeFiniteRandomInputs(500, 2468u);
  input.insert(input.end(), finite_random.begin(), finite_random.end());
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

// Direct semantic coverage for IEEE/ONNX special values at the smallest
// possible buffer size (N=1) -- the "scalar tail" extreme, and exactly the
// shape used by single-lookup RNN gate activations (see
// providers/cpu/rnn/rnn_helpers.cc). Calls MlasTanhKernelAppleAccelerate
// directly and asserts the operator's own contract via
// ExpectIeeeTanhSemantics, independent of the portable kernel.
TEST(TanhAppleAccelerate, SpecialValuesSemanticsSingleElement) {
  for (float value : MakeSpecialValueInputs()) {
    float output = 0.0f;
    MlasTanhKernelAppleAccelerate(&value, &output, 1);
    ExpectIeeeTanhSemantics(output, value);
  }
}

// Direct semantic coverage for IEEE/ONNX special values at a buffer size
// (N=3) that is not a multiple of the portable kernel's SIMD width (4), so
// it exercises the same "scalar-tail" buffer shapes the portable reference
// kernel documents as its only reliable path for NaN handling -- except
// here the Apple kernel's output is checked directly against the operator
// contract, not against the portable kernel. Each special value is placed
// at the middle of three elements, with ordinary finite neighbors on either
// side, to also catch a kernel bug where a special value corrupts adjacent
// lanes (e.g. an indexing or chunking off-by-one).
TEST(TanhAppleAccelerate, SpecialValuesSemanticsScalarTailBuffer) {
  constexpr float kLeftNeighbor = 1.25f;
  constexpr float kRightNeighbor = -3.5f;

  for (float special_value : MakeSpecialValueInputs()) {
    std::array<float, 3> input = {kLeftNeighbor, special_value, kRightNeighbor};
    std::array<float, 3> apple_output = {0.0f, 0.0f, 0.0f};
    MlasTanhKernelAppleAccelerate(input.data(), apple_output.data(), input.size());

    ExpectIeeeTanhSemantics(apple_output[1], input[1]);

    // The ordinary finite neighbors must still match the portable baseline,
    // confirming the special value did not corrupt adjacent lanes.
    std::array<float, 3> portable_output = {0.0f, 0.0f, 0.0f};
    MlasTanhKernel(input.data(), portable_output.data(), input.size());
    ExpectMatchesPortableKernel(apple_output[0], portable_output[0], input[0]);
    ExpectMatchesPortableKernel(apple_output[2], portable_output[2], input[2]);
  }
}

#endif  // defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)
