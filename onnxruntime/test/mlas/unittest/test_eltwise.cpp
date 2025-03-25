// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/eltwise.h"

class MlasEltwiseAddTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInputLeft;
  MatrixGuardBuffer<float> BufferInputRight;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MatrixGuardBuffer<MLAS_FP16> BufferInputLeftFp16;
  MatrixGuardBuffer<MLAS_FP16> BufferInputRightFp16;
  MatrixGuardBuffer<MLAS_FP16> BufferOutputFp16;

  void Test(size_t N, float MinimumValue, float MaximumValue, const std::optional<float>& ScalarValue = std::nullopt) {
    float* InputLeft = BufferInputLeft.GetBuffer(N);
    float* InputRight = BufferInputRight.GetBuffer(N);
    float* Output = BufferOutput.GetBuffer(N);
    float* OutputReference = BufferOutputReference.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      InputLeft[n] = distribution(generator);
      InputRight[n] = ScalarValue.value_or(distribution(generator));
    }

    for (size_t n = 0; n < N; n++) {
      OutputReference[n] = InputLeft[n] + InputRight[n];
    }

    MlasEltwiseAdd(InputLeft, InputRight, Output, N);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    for (size_t n = 0; n < N; n++) {
      float diff = std::fabs(Output[n] - OutputReference[n]);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(OutputReference[n]) * RelativeTolerance)
          << " @" << n << " of " << N << ", got: " << Output[n] << ", expecting: " << OutputReference[n];
    }
  }

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

  void TestFp16(size_t N, float MinimumValue, float MaximumValue, const std::optional<float>& ScalarValue = std::nullopt) {
    MLAS_FP16* InputLeft = BufferInputLeftFp16.GetBuffer(N);
    MLAS_FP16* InputRight = BufferInputRightFp16.GetBuffer(N);
    MLAS_FP16* Output = BufferOutputFp16.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      InputLeft[n] = MLAS_FP16(distribution(generator));
      InputRight[n] = MLAS_FP16(ScalarValue.value_or(distribution(generator)));
    }

    MlasEltwiseAdd(InputLeft, InputRight, Output, N);

    constexpr float AbsoluteTolerance = 5e-4f;
    constexpr float RelativeTolerance = 1e-3f;

    for (size_t n = 0; n < N; n++) {
      float inLeft = InputLeft[n].ToFloat();
      float inRight = InputRight[n].ToFloat();
      float ref = inLeft + inRight;
      float out = Output[n].ToFloat();
      float diff = std::fabs(out - ref);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
          << " @ " << inLeft << ", " << inRight << ", got: " << out << ", expecting: " << ref
          << ", r-diff: " << diff / std::fabs(ref);
    }
  }

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Eltwise_Add");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n < 128; n++) {
      Test(n, -10.f, 10.f);
      Test(n, -10.f, 10.f, -5000.f);
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
      TestFp16(n, -17.f, 11.f);
      TestFp16(n, -17.f, 11.f, -5000.f);
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasEltwiseAddTest>::RegisterShortExecute();
  }
  return count;
});
