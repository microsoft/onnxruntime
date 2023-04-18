// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

template <typename xint8_t>
class MlasQuantizeLinearTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<xint8_t> BufferOutput;
  MatrixGuardBuffer<xint8_t> BufferOutputReference;

  void GenerateReference(const float* Input, xint8_t* OutputReference, size_t N, float Scale, xint8_t ZeroPoint) {
    for (size_t n = 0; n < N; n++) {
      float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
      FloatValue = std::max(FloatValue, float(std::numeric_limits<xint8_t>::min()));
      FloatValue = std::min(FloatValue, float(std::numeric_limits<xint8_t>::max()));
      OutputReference[n] = (xint8_t)FloatValue;
    }
  }

  void Test(size_t N) {
    float* Input = BufferInput.GetBuffer(N);
    xint8_t* Output = BufferOutput.GetBuffer(N);
    xint8_t* OutputReference = BufferOutputReference.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));

    std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
    float MinimumValue = min_gen(generator);

    std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
    float MaximumValue = max_gen(generator);

    float Scale = (MaximumValue - MinimumValue) / 512.f;

    std::uniform_int_distribution<int32_t> zp_distribution(std::numeric_limits<xint8_t>::min(), std::numeric_limits<xint8_t>::max());
    xint8_t ZeroPoint = static_cast<xint8_t>(zp_distribution(generator));

    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);
    for (size_t n = 0; n < N; n++) {
      Input[n] = distribution(generator);
    }

    GenerateReference(Input, OutputReference, N, Scale, ZeroPoint);
    MlasQuantizeLinear(Input, Output, N, Scale, ZeroPoint);

    for (size_t n = 0; n < N; n++) {
      ASSERT_EQ(Output[n], OutputReference[n]) << ", size=" << N << ", index=" << n;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name(std::is_signed<xint8_t>::value ? "QuantizeLinearS8" : "QuantizeLinearU8");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n <= 512; n++) {
      Test(n);
    }
  }
};

template <>
MlasQuantizeLinearTest<int8_t>* MlasTestFixture<MlasQuantizeLinearTest<int8_t>>::mlas_tester(nullptr);
template <>
MlasQuantizeLinearTest<uint8_t>* MlasTestFixture<MlasQuantizeLinearTest<uint8_t>>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<int8_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<uint8_t>>::RegisterShortExecute();
  }
  return count;
});
