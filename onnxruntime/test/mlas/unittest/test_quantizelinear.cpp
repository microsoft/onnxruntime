// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

template <typename QuantInt>
class MlasQuantizeLinearTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<QuantInt> BufferOutput;
  MatrixGuardBuffer<QuantInt> BufferOutputReference;

  void GenerateReference(const float* Input, QuantInt* OutputReference, size_t N, float Scale, QuantInt ZeroPoint) {
    for (size_t n = 0; n < N; n++) {
      float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
      FloatValue = std::max(FloatValue, static_cast<float>(std::numeric_limits<QuantInt>::min()));
      FloatValue = std::min(FloatValue, static_cast<float>(std::numeric_limits<QuantInt>::max()));
      OutputReference[n] = static_cast<QuantInt>(FloatValue);
    }
  }

  void Test(size_t N) {
    float* Input = BufferInput.GetBuffer(N);
    QuantInt* Output = BufferOutput.GetBuffer(N);
    QuantInt* OutputReference = BufferOutputReference.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));

    std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
    float MinimumValue = min_gen(generator);

    std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
    float MaximumValue = max_gen(generator);

    float Scale = (MaximumValue - MinimumValue) / 512.f;

    std::uniform_int_distribution<int32_t> zp_distribution(std::numeric_limits<QuantInt>::min(),
                                                           std::numeric_limits<QuantInt>::max());
    QuantInt ZeroPoint = static_cast<QuantInt>(zp_distribution(generator));

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
    if constexpr (std::is_same_v<QuantInt, int8_t>) {
      return "QuantizeLinearS8";
    } else if (std::is_same_v<QuantInt, uint8_t>) {
      return "QuantizeLinearU8";
    } else if (std::is_same_v<QuantInt, int16_t>) {
      return "QuantizeLinearS16";
    } else {
      return "QuantizeLinearU16";
    }
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
template <>
MlasQuantizeLinearTest<int16_t>* MlasTestFixture<MlasQuantizeLinearTest<int16_t>>::mlas_tester(nullptr);
template <>
MlasQuantizeLinearTest<uint16_t>* MlasTestFixture<MlasQuantizeLinearTest<uint16_t>>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<int8_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<uint8_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<int16_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<uint16_t>>::RegisterShortExecute();
  }
  return count;
});
