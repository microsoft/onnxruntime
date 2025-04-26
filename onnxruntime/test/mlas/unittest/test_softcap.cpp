// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/softmax.h"

class MlasComputeTanhTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<MLAS_FP16> BufferInputFp16;
  MatrixGuardBuffer<MLAS_FP16> BufferOutputFp16;

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
  void TestFp16(size_t N, float MinimumValue, float MaximumValue) {
    MLAS_FP16* Input = BufferInputFp16.GetBuffer(N);
    MLAS_FP16* Output = BufferOutputFp16.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      Input[n] = MLAS_FP16(distribution(generator));
    }

    MlasComputeTanh(Input, Output, N);

    constexpr float AbsoluteTolerance = 5e-3f;
    constexpr float RelativeTolerance = 5e-3f;

    for (size_t n = 0; n < N; n++) {
      float in = Input[n].ToFloat();
      float ref = std::tanh(in);
      float out = Output[n].ToFloat();
      float diff = std::fabs(out - ref);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
          << " @ " << in << ", got: " << out << ", expecting: " << ref
          << ", diff: " << diff << ", r-diff: " << diff / std::fabs(ref);
    }
  }
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Tanh");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n < 128; n++) {
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
      TestFp16(n, -3.51562f, 3.51562f);
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    }
  }
};

class MlasComputeSoftcapTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<MLAS_FP16> BufferInputFp16;
  MatrixGuardBuffer<MLAS_FP16> BufferOutputFp16;

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
  void TestFp16(size_t N, float MinimumValue, float MaximumValue, float cap) {
    MLAS_FP16* Input = BufferInputFp16.GetBuffer(N);
    MLAS_FP16* Output = BufferOutputFp16.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      Input[n] = MLAS_FP16(distribution(generator));
    }

    MlasComputeSoftcap(Input, Output, N, MLAS_FP16(cap));

    constexpr float AbsoluteTolerance = 5e-3f;
    constexpr float RelativeTolerance = 5e-3f;

    for (size_t n = 0; n < N; n++) {
      float in = Input[n].ToFloat();
      float ref = std::tanh(in / cap) * cap;
      float out = Output[n].ToFloat();
      float diff = std::fabs(out - ref);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
          << " @ " << in << ", got: " << out << ", expecting: " << ref << ", r-diff " << diff / std::fabs(ref);
    }
  }
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Softcap");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n < 128; n++) {
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
      TestFp16(n, -10.f, 10.f, 3.2f);
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasComputeTanhTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasComputeSoftcapTest>::RegisterShortExecute();
  }
  return count;
});
