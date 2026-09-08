// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include "test_util.h"
#include "core/mlas/lib/mlasi.h"

// Exercises MlasComputeErf. On AVX-512 this dispatches to the 16-wide
// MlasErfKernelAvx512F (added for the MobileClip GELU path); on other targets it
// uses the FMA3/scalar kernel.
//
// Two levels of verification:
//   1. Accuracy-neutrality vs base: on AVX-512 hardware, compare the new
//      MlasErfKernelAvx512F directly against the base MlasErfKernelFma3 kernel
//      (the pointer this change replaced). This proves "matches base", which is
//      the actual guarantee -- not merely "matches the math".
//   2. Mathematical correctness: MlasComputeErf vs std::erf within the
//      polynomial's accuracy tolerance, sweeping lengths that straddle the
//      16-lane boundary so the vector main loop and masked tail are both hit.
class MlasErfTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputBase;

  // MLAS erf is a minimax polynomial approximation; empirically its absolute
  // error vs std::erf stays well under 1e-5. Conservative bound guards the
  // kernel (width/dispatch bugs) without being flaky on the approximation.
  static constexpr float AbsTolerance = 2e-5f;

#if defined(MLAS_TARGET_AMD64)
  static bool Avx512Available() {
    return GetMlasPlatform().ErfKernelRoutine == MlasErfKernelAvx512F;
  }
#endif  // MLAS_TARGET_AMD64

  static uint32_t Bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
  }

  // ULP distance between two finite floats of the same sign convention.
  static uint32_t UlpDiff(float a, float b) {
    if (a == b) return 0;
    int32_t ia = static_cast<int32_t>(Bits(a));
    int32_t ib = static_cast<int32_t>(Bits(b));
    // Map to a monotonic ordering across the sign boundary.
    if (ia < 0) ia = static_cast<int32_t>(0x80000000u) - ia;
    if (ib < 0) ib = static_cast<int32_t>(0x80000000u) - ib;
    int64_t d = int64_t(ia) - int64_t(ib);
    return static_cast<uint32_t>(d < 0 ? -d : d);
  }

#if defined(MLAS_TARGET_AMD64)
  // Level 1: AVX-512 kernel must match the base FMA3 kernel to <= 1 ULP.
  // (The two implementations share the erf polynomial coefficients but use
  // different instruction sequences -- hand-written AVX2 asm vs AVX-512
  // intrinsics -- so FMA reassociation could in principle differ by the last
  // bit. Measured divergence on current hardware is 0 ULP, i.e. bit-exact; the
  // 1 ULP bound is retained as the safe contract across microarchitectures.)
  void TestMatchesBase(size_t N) {
    if (!Avx512Available()) {
      return;  // Base FMA3 kernel is what runs anyway; nothing to compare.
    }

    float* Input = BufferInput.GetBuffer(N);
    float* Opt = BufferOutput.GetBuffer(N);
    float* Base = BufferOutputBase.GetBuffer(N);

    for (size_t i = 0; i < N; i++) {
      Input[i] = -6.0f + 12.0f * (static_cast<float>(i % 257) / 256.0f);
    }

    MlasErfKernelAvx512F(Input, Opt, N);
    MlasErfKernelFma3(Input, Base, N);

    uint32_t max_ulp = 0;
    for (size_t i = 0; i < N; i++) {
      max_ulp = std::max(max_ulp, UlpDiff(Opt[i], Base[i]));
    }
    ASSERT_LE(max_ulp, 1u) << " AVX-512 erf diverges from base FMA3 by " << max_ulp
                           << " ULP at N=" << N;
  }

  // Level 1 for edge inputs: NaN, +/-inf, denormals, large saturating magnitudes.
  // erf saturates to +/-1 outside ~[-4, 4]; NaN must propagate.
  void TestSpecialValuesMatchBase() {
    if (!Avx512Available()) {
      return;
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float denorm = std::numeric_limits<float>::denorm_min();
    const std::vector<float> specials = {
        nan, -nan, inf, -inf, denorm, -denorm,
        0.0f, -0.0f, 1e-30f, -1e-30f, 10.0f, -10.0f, 1e30f, -1e30f,
        3.9f, -3.9f, 4.1f, -4.1f};
    const size_t N = specials.size();

    float* Input = BufferInput.GetBuffer(N);
    float* Opt = BufferOutput.GetBuffer(N);
    float* Base = BufferOutputBase.GetBuffer(N);
    std::copy(specials.begin(), specials.end(), Input);

    MlasErfKernelAvx512F(Input, Opt, N);
    MlasErfKernelFma3(Input, Base, N);

    for (size_t i = 0; i < N; i++) {
      if (std::isnan(Base[i])) {
        ASSERT_TRUE(std::isnan(Opt[i])) << " expected NaN at i=" << i << ", input=" << Input[i];
      } else {
        ASSERT_LE(UlpDiff(Opt[i], Base[i]), 1u)
            << " special value mismatch at i=" << i << ", input=" << Input[i]
            << ", opt=" << Opt[i] << ", base=" << Base[i];
      }
    }
  }
#endif  // MLAS_TARGET_AMD64

  // Level 2: mathematical correctness vs std::erf, in place (matches the
  // MobileClip GELU usage where the activation updates the tensor in place).
  void TestMathInPlace(size_t N) {
    float* Buffer = BufferInput.GetBuffer(N);
    std::vector<float> Reference(N);

    for (size_t i = 0; i < N; i++) {
      Buffer[i] = -4.0f + 8.0f * (static_cast<float>(i % 101) / 100.0f);
      Reference[i] = std::erf(Buffer[i]);
    }

    MlasComputeErf(Buffer, Buffer, N);

    for (size_t i = 0; i < N; i++) {
      ASSERT_NEAR(Buffer[i], Reference[i], AbsTolerance) << " N=" << N << ", i=" << i;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Erf");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    // Lengths crossing the 16-lane boundary: exact multiples, one-past, and odd
    // tails exercise both the vector main loop and the masked remainder.
    for (size_t n : {size_t(1), size_t(3), size_t(7), size_t(15), size_t(16),
                     size_t(17), size_t(31), size_t(32), size_t(33), size_t(48),
                     size_t(63), size_t(64), size_t(255), size_t(1000)}) {
#if defined(MLAS_TARGET_AMD64)
      TestMatchesBase(n);
#endif
      TestMathInPlace(n);
    }
#if defined(MLAS_TARGET_AMD64)
    TestSpecialValuesMatchBase();
#endif
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasErfTest>::RegisterShortExecute() : 0;
});
