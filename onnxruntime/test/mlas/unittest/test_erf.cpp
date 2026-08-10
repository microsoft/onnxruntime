// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include "test_util.h"

// Exercises MlasComputeErf. On AVX-512 this dispatches to the 16-wide
// MlasErfKernelAvx512F (added for the MobileClip GELU path); on other targets it
// uses the FMA3/scalar kernel. The test asserts the result matches the C++
// library std::erf within the MLAS polynomial's accuracy tolerance, and sweeps
// buffer lengths that straddle the 16-lane boundary so the vector main loop and
// the masked-tail path are both covered.
class MlasErfTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;

  // MLAS erf is a minimax polynomial approximation; empirically its absolute
  // error vs std::erf stays well under 1e-5. Use a conservative bound so the
  // test guards the kernel (width/dispatch bugs) without being flaky on the
  // approximation itself.
  static constexpr float AbsTolerance = 2e-5f;

  void Test(size_t N) {
    float* Input = BufferInput.GetBuffer(N);
    float* Output = BufferOutput.GetBuffer(N);

    // Deterministic spread across the interesting erf range [-4, 4], where the
    // approximation transitions between its small/large branches and saturates.
    for (size_t i = 0; i < N; i++) {
      Input[i] = -4.0f + 8.0f * (static_cast<float>(i % 101) / 100.0f);
    }

    MlasComputeErf(Input, Output, N);

    for (size_t i = 0; i < N; i++) {
      const float expected = std::erf(Input[i]);
      const float actual = Output[i];
      ASSERT_NEAR(actual, expected, AbsTolerance)
          << " N=" << N << ", i=" << i << ", input=" << Input[i];
    }
  }

  // Verify in-place operation (Input aliased as Output) matches the MobileClip
  // usage, where the activation updates the tensor in place.
  void TestInPlace(size_t N) {
    float* Buffer = BufferInput.GetBuffer(N);
    std::vector<float> Reference(N);

    for (size_t i = 0; i < N; i++) {
      Buffer[i] = -3.0f + 6.0f * (static_cast<float>(i % 97) / 96.0f);
      Reference[i] = std::erf(Buffer[i]);
    }

    MlasComputeErf(Buffer, Buffer, N);

    for (size_t i = 0; i < N; i++) {
      ASSERT_NEAR(Buffer[i], Reference[i], AbsTolerance)
          << " [in-place] N=" << N << ", i=" << i;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Erf");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    // Lengths crossing the 16-lane boundary: exact multiples, one-past, and
    // odd tails exercise both the vector main loop and the masked remainder.
    for (size_t n : {size_t(1), size_t(3), size_t(7), size_t(15), size_t(16),
                     size_t(17), size_t(31), size_t(32), size_t(33), size_t(48),
                     size_t(63), size_t(64), size_t(255), size_t(1000)}) {
      Test(n);
      TestInPlace(n);
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasErfTest>::RegisterShortExecute() : 0;
});
