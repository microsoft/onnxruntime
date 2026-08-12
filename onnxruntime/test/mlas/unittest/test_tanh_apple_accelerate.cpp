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
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 1e-6f;
constexpr float kRelativeTolerance = 1e-6f;

// Poison value used to pre-fill output buffers before calling a kernel
// under test, instead of the natural-looking 0.0f. A kernel that silently
// fails to write an element (e.g. an off-by-one in the scalar-tail path)
// would otherwise leave a 0.0f-initialized buffer holding a value that,
// for several of these special/denormal/near-zero test inputs, is
// indistinguishable from a genuinely correct answer (tanh(+/-0) == +/-0,
// and an FTZ-flushed near-zero result may also be exactly 0.0f) -- so a
// no-write bug could pass silently.
//
// This is deliberately a FINITE, out-of-range sentinel (not NaN). tanh's
// real range is [-1, 1] for every finite/infinite input, so any finite
// magnitude far outside that range can never be a genuinely correct output
// and is guaranteed to fail loudly if a kernel leaves it unwritten. A NaN
// poison value was tried first and had a real no-write hole: for the two
// NaN *inputs* in MakeSpecialValueInputs (quiet_NaN, signaling_NaN),
// ExpectIeeeTanhSemantics's own contract requires NaN *output*, so an
// unwritten NaN-poisoned buffer would satisfy std::isnan(value) and the
// test would pass even though the kernel never wrote anything -- silently
// defeating the entire poison mechanism for exactly the inputs it most
// needs to guard. This finite sentinel has no such blind spot: it is
// unequal to every expected output (NaN, +/-1.0f, +/-0.0f, or any
// magnitude bounded by the input for the denormal/near-zero cases).
constexpr float kPoisonValue = 123456.75f;

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

// Smallest-magnitude subnormal (denormal) inputs only, in both signs.
// FLT_MIN (the smallest positive *normal* float) is deliberately NOT
// included here -- see MakeSmallestNormalInputs and
// ExpectSmallestNormalSemantics below. These probe two divergent behaviors
// real hardware/libm implementations are free to choose between for
// subnormal inputs: an IEEE-correctly-rounded tanh(x) that rounds to
// exactly x (the -x^3/3 correction term is unrepresentably smaller than
// one ULP at these magnitudes), or a flush-to-zero (FTZ) implementation
// that treats a subnormal input/result as zero. Asserted via
// ExpectDenormalCompatibleSemantics, which intentionally only checks the
// invariants true under EITHER behavior -- see that function's comment.
std::vector<float> MakeDenormalInputs() {
  return {
      std::numeric_limits<float>::denorm_min(),   // smallest positive subnormal
      -std::numeric_limits<float>::denorm_min(),  // smallest negative subnormal
  };
}

// +/-FLT_MIN: the smallest positive *normal* float (and its negation). This
// is intentionally kept separate from MakeDenormalInputs -- FLT_MIN is not
// a subnormal value, so unlike true denormals, an implementation has no
// FTZ (flush-to-zero) license here: FTZ modes flush subnormal inputs and
// results, not normal ones, so a compliant implementation must not zero
// this input or its result. tanh(x) evaluated in exact real arithmetic at
// x = FLT_MIN is FLT_MIN - FLT_MIN^3/3 + O(FLT_MIN^5); FLT_MIN^3 is itself
// far below FLT_MIN's own ULP (roughly 2^-126 vs. an ULP of roughly
// 2^-149 at this magnitude relative to the correction term's exponent, so
// the correction is many orders of magnitude smaller than one ULP of
// FLT_MIN), so the correctly-rounded float32 result is exactly FLT_MIN.
// Asserted strictly via ExpectSmallestNormalSemantics (exact value, sign
// preserved), not the either/or FTZ-tolerant assertion used for true
// denormals.
std::vector<float> MakeSmallestNormalInputs() {
  return {
      std::numeric_limits<float>::min(),   // smallest positive normal (FLT_MIN)
      -std::numeric_limits<float>::min(),  // smallest negative normal (FLT_MIN)
  };
}

// Asserts only the invariants that hold under BOTH plausible hardware
// behaviors for a TRUE DENORMAL (subnormal) input -- see MakeDenormalInputs.
// Do NOT use this for +/-FLT_MIN (a normal value): see
// ExpectSmallestNormalSemantics, which asserts strict/exact semantics
// instead, since flush-to-zero is not a legitimate rationale for a normal
// input.
//   * IEEE-correctly-rounded: tanh(x) rounds to exactly x for |x| this
//     small, so output == input (magnitude unchanged, sign preserved) is a
//     valid outcome.
//   * Flush-to-zero (FTZ): a fast-math/vector-unit implementation may treat
//     a subnormal input (or a subnormal result) as zero. Some FTZ
//     implementations preserve the sign of the flushed-to-zero result and
//     some do not (unlike the exact +/-0.0 special-value contract asserted
//     by ExpectIeeeTanhSemantics, this is underflow/rounding behavior, not
//     part of the operator's own IEEE contract), so output == 0.0f of
//     EITHER sign is a valid outcome here.
// We deliberately do NOT assert which of these vForce actually does on
// real Apple Silicon -- that would be inventing unverified hardware
// behavior, which no Apple hardware is available in this environment to
// confirm. Instead this asserts only what both behaviors share: never NaN,
// sign preserved for a genuinely nonzero result, and magnitude never
// amplified beyond the input's. The actual value is printed (not merely
// asserted past) so a run on real hardware makes which behavior applies
// observable.
void ExpectDenormalCompatibleSemantics(float value, float input) {
  EXPECT_FALSE(std::isnan(value)) << "input=" << input << " (denormal/near-zero), got NaN output";
  if (value != 0.0f) {
    // Sign is only required to match when the result is nonzero. A
    // flush-to-zero-to-0.0f result is permitted to carry either sign (see
    // comment above), so a zero result is deliberately excluded from this
    // check rather than requiring it to match the input's sign.
    EXPECT_EQ(std::signbit(value), std::signbit(input))
        << "input=" << input << ", expected sign preserved for a nonzero result (IEEE-exact "
        << "tanh(x)~=x is odd), got signbit=" << std::signbit(value);
  }
  EXPECT_LE(std::fabs(value), std::fabs(input))
      << "input=" << input << ", expected |output| <= |input| (both IEEE-exact tanh(x)~=x and "
      << "flush-to-zero satisfy this), got |output|=" << std::fabs(value);
  // Not an assertion -- recorded so a human running this on real hardware
  // can see which of the two behaviors (IEEE-exact passthrough vs.
  // flush-to-zero, and which sign convention if flushed to zero) this
  // build/hardware actually exhibits.
  std::cerr << "[ TanhAppleAccelerate denormal observation ] input=" << input << " output=" << value << "\n";
}

// Asserts strict, exact semantics for +/-FLT_MIN (the smallest positive
// normal float and its negation) -- see MakeSmallestNormalInputs for why
// this is NOT the FTZ-tolerant either/or check used for true denormals.
// FLT_MIN is a normal value, so a flush-to-zero rationale does not apply:
// the correctly-rounded float32 tanh(FLT_MIN) is exactly FLT_MIN (the
// -x^3/3 correction term is many orders of magnitude below one ULP of
// FLT_MIN), so this requires exact equality and sign preservation, not a
// tolerance.
void ExpectSmallestNormalSemantics(float value, float input) {
  EXPECT_EQ(value, input) << "input=" << input << " (+/-FLT_MIN, smallest normal float -- not a denormal, so "
                          << "flush-to-zero is not a valid rationale here), expected output to round to "
                          << "exactly the input, got " << value;
  EXPECT_EQ(std::signbit(value), std::signbit(input))
      << "input=" << input << " (+/-FLT_MIN), expected sign preserved, got signbit=" << std::signbit(value);
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
    // Poison-initialized (not 0.0f): tanh(+/-0) == +/-0, so a 0.0f-init
    // buffer could not distinguish "kernel wrote the correct answer" from
    // "kernel never wrote anything". See kPoisonValue's comment.
    float output = kPoisonValue;
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
    // Poison-initialized (not 0.0f) for the same reason as the
    // single-element test above: tanh(+/-0) == +/-0, so a 0.0f-init
    // buffer could mask a kernel that never wrote the middle element.
    std::array<float, 3> apple_output = {kPoisonValue, kPoisonValue, kPoisonValue};
    MlasTanhKernelAppleAccelerate(input.data(), apple_output.data(), input.size());

    ExpectIeeeTanhSemantics(apple_output[1], input[1]);

    // The ordinary finite neighbors must still match the portable baseline,
    // confirming the special value did not corrupt adjacent lanes.
    std::array<float, 3> portable_output = {kPoisonValue, kPoisonValue, kPoisonValue};
    MlasTanhKernel(input.data(), portable_output.data(), input.size());
    ExpectMatchesPortableKernel(apple_output[0], portable_output[0], input[0]);
    ExpectMatchesPortableKernel(apple_output[2], portable_output[2], input[2]);
  }
}

// Direct semantic coverage for TRUE DENORMAL (subnormal) inputs at the
// smallest possible buffer size (N=1). Unlike the NaN/Inf/signed-zero cases
// above, real hardware/libm implementations are free to choose between
// IEEE-exact tanh(x)~=x passthrough and flush-to-zero (FTZ) for values this
// tiny, so this only asserts the invariants both behaviors share (see
// ExpectDenormalCompatibleSemantics); it does not assert a specific vForce
// behavior we have not observed on real Apple Silicon. +/-FLT_MIN (a
// normal, not subnormal, value) is intentionally NOT covered by this test
// -- see SmallestNormalSemanticsSingleElement below, which asserts strict
// semantics instead since FTZ is not a valid rationale for a normal input.
TEST(TanhAppleAccelerate, DenormalAndNearZeroSemanticsSingleElement) {
  for (float value : MakeDenormalInputs()) {
    // Poison-initialized (not 0.0f): a flush-to-zero result is a valid
    // outcome here (see ExpectDenormalCompatibleSemantics), so a 0.0f-init
    // buffer could not distinguish a real FTZ-to-zero result from the
    // kernel never having written anything.
    float output = kPoisonValue;
    MlasTanhKernelAppleAccelerate(&value, &output, 1);
    ExpectDenormalCompatibleSemantics(output, value);
  }
}

// Direct semantic coverage for +/-FLT_MIN (the smallest positive normal
// float) at the smallest possible buffer size (N=1). Deliberately separate
// from the true-denormal test above: FLT_MIN is a normal value, so unlike
// true subnormals, a flush-to-zero rationale does not apply, and this
// asserts strict/exact semantics via ExpectSmallestNormalSemantics instead
// of the either/or FTZ-tolerant check.
TEST(TanhAppleAccelerate, SmallestNormalSemanticsSingleElement) {
  for (float value : MakeSmallestNormalInputs()) {
    // Poison-initialized (not 0.0f) for the same no-write-detection reason
    // as the other single-element tests above.
    float output = kPoisonValue;
    MlasTanhKernelAppleAccelerate(&value, &output, 1);
    ExpectSmallestNormalSemantics(output, value);
  }
}

// Same denormal/near-zero coverage at a buffer size (N=3) not a multiple of
// the portable kernel's SIMD width, with the denormal/near-zero value
// flanked by ordinary finite neighbors, to also catch a kernel bug where a
// tiny-magnitude value corrupts adjacent lanes (e.g. an indexing or
// chunking off-by-one, or an FTZ mode applied to the whole vector register
// rather than per-lane).
TEST(TanhAppleAccelerate, DenormalAndNearZeroSemanticsScalarTailBuffer) {
  constexpr float kLeftNeighbor = 1.25f;
  constexpr float kRightNeighbor = -3.5f;

  for (float denormal_value : MakeDenormalInputs()) {
    std::array<float, 3> input = {kLeftNeighbor, denormal_value, kRightNeighbor};
    // Poison-initialized (not 0.0f) for the same reason as the
    // single-element test above: a flush-to-zero result is a valid
    // outcome, so a 0.0f-init buffer could mask a kernel that never wrote
    // the middle element.
    std::array<float, 3> apple_output = {kPoisonValue, kPoisonValue, kPoisonValue};
    MlasTanhKernelAppleAccelerate(input.data(), apple_output.data(), input.size());

    ExpectDenormalCompatibleSemantics(apple_output[1], input[1]);

    // The ordinary finite neighbors must still match the portable
    // baseline, confirming the denormal/near-zero value did not corrupt
    // adjacent lanes.
    std::array<float, 3> portable_output = {kPoisonValue, kPoisonValue, kPoisonValue};
    MlasTanhKernel(input.data(), portable_output.data(), input.size());
    ExpectMatchesPortableKernel(apple_output[0], portable_output[0], input[0]);
    ExpectMatchesPortableKernel(apple_output[2], portable_output[2], input[2]);
  }
}

// Same +/-FLT_MIN coverage as SmallestNormalSemanticsSingleElement above,
// at a buffer size (N=3) not a multiple of the portable kernel's SIMD
// width, flanked by ordinary finite neighbors, to also catch a kernel bug
// where this value corrupts adjacent lanes.
TEST(TanhAppleAccelerate, SmallestNormalSemanticsScalarTailBuffer) {
  constexpr float kLeftNeighbor = 1.25f;
  constexpr float kRightNeighbor = -3.5f;

  for (float smallest_normal_value : MakeSmallestNormalInputs()) {
    std::array<float, 3> input = {kLeftNeighbor, smallest_normal_value, kRightNeighbor};
    std::array<float, 3> apple_output = {kPoisonValue, kPoisonValue, kPoisonValue};
    MlasTanhKernelAppleAccelerate(input.data(), apple_output.data(), input.size());

    ExpectSmallestNormalSemantics(apple_output[1], input[1]);

    // The ordinary finite neighbors must still match the portable
    // baseline, confirming this value did not corrupt adjacent lanes.
    std::array<float, 3> portable_output = {kPoisonValue, kPoisonValue, kPoisonValue};
    MlasTanhKernel(input.data(), portable_output.data(), input.size());
    ExpectMatchesPortableKernel(apple_output[0], portable_output[0], input[0]);
    ExpectMatchesPortableKernel(apple_output[2], portable_output[2], input[2]);
  }
}

#endif  // defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)
