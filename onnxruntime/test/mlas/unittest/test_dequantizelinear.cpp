// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

template <typename QuantInt>
class MlasDequantizeLinearTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<QuantInt> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;

  void GenerateReference(const QuantInt* Input, float* OutputReference, size_t N, float Scale, QuantInt ZeroPoint) {
    int32_t ZeroPointS32 = static_cast<int32_t>(ZeroPoint);

    for (size_t n = 0; n < N; n++) {
      OutputReference[n] = static_cast<float>(static_cast<int32_t>(Input[n]) - ZeroPointS32) * Scale;
    }
  }

  void Test(size_t N) {
    QuantInt* Input = BufferInput.GetBuffer(N);
    float* Output = BufferOutput.GetBuffer(N);
    float* OutputReference = BufferOutputReference.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));

    std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
    float MinimumValue = min_gen(generator);

    std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
    float MaximumValue = max_gen(generator);

    float Scale = (MaximumValue - MinimumValue) / 512.f;

    std::uniform_int_distribution<int32_t> zp_distribution(std::numeric_limits<QuantInt>::min(),
                                                           std::numeric_limits<QuantInt>::max());
    QuantInt ZeroPoint = static_cast<QuantInt>(zp_distribution(generator));

    for (size_t n = 0; n < N; n++) {
      Input[n] = static_cast<QuantInt>(zp_distribution(generator));
    }

    GenerateReference(Input, OutputReference, N, Scale, ZeroPoint);
    MlasDequantizeLinear(Input, Output, N, Scale, ZeroPoint);

    for (size_t n = 0; n < N; n++) {
      ASSERT_EQ(Output[n], OutputReference[n]) << ", size=" << N << ", index=" << n;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    if constexpr (std::is_same_v<QuantInt, int8_t>) {
      return "DequantizeLinearS8";
    } else {
      return "DequantizeLinearU8";
    }
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n <= 512; n++) {
      Test(n);
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasDequantizeLinearTest<int8_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasDequantizeLinearTest<uint8_t>>::RegisterShortExecute();
  }
  return count;
});
