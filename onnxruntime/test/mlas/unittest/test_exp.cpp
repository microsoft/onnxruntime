// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasComputeExpTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;

  void Test(size_t N, float MinimumValue, float MaximumValue) {
    float* Input = BufferInput.GetBuffer(N);
    float* Output = BufferOutput.GetBuffer(N);
    float* OutputReference = BufferOutputReference.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      Input[n] = distribution(generator);
    }

    for (size_t n = 0; n < N; n++) {
      OutputReference[n] = std::exp(Input[n]);
    }

    MlasComputeExp(Input, Output, N);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    for (size_t n = 0; n < N; n++) {
      float diff = std::fabs(Output[n] - OutputReference[n]);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(OutputReference[n]) * RelativeTolerance)
          << " @" << n << " of " << N << ", got: " << Output[n] << ", expecting: " << OutputReference[n];
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Exp");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n < 128; n++) {
      Test(n, -10.f, 10.f);
    }
  }
};

template <> MlasComputeExpTest* MlasTestFixture<MlasComputeExpTest>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  // no long execute needed
  return is_short_execute ? MlasDirectShortExecuteTests<MlasComputeExpTest>::RegisterShortExecute() : 0;
});
