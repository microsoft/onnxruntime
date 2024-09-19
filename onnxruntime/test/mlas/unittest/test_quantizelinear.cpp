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

template <bool Signed>
class MlasQuantizeLinear4BitTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<uint8_t> BufferOutput;
  MatrixGuardBuffer<uint8_t> BufferOutputReference;

  int32_t MinVal() const {
    if constexpr (Signed) {
      return -8;
    } else {
      return 0;
    }
  }

  int32_t MaxVal() const {
    if constexpr (Signed) {
      return 7;
    } else {
      return 15;
    }
  }

  void GenerateReference(const float* Input, uint8_t* OutputReference, size_t N, float Scale,
                         int8_t ZeroPoint) {
    for (size_t n = 0; n < N; n++) {
      float FloatValue = std::nearbyintf(Input[n] / Scale) + static_cast<float>(ZeroPoint);
      FloatValue = std::max(FloatValue, static_cast<float>(MinVal()));
      FloatValue = std::min(FloatValue, static_cast<float>(MaxVal()));

      int8_t IntValue = static_cast<int8_t>(FloatValue);

      size_t i = n >> 1;
      size_t j = n & 0x1;
      uint8_t Shift = 4 * static_cast<uint8_t>(j);
      uint8_t Mask = 0xF << Shift;

      OutputReference[i] &= ~Mask;                                            // Clear 4-bit lane
      OutputReference[i] |= static_cast<uint8_t>((IntValue & 0xF) << Shift);  // Set 4-bit lane
    }
  }

  void Test(size_t N) {
    size_t OutBufLen = (N + 1) / 2;
    float* Input = BufferInput.GetBuffer(N);
    uint8_t* Output = BufferOutput.GetBuffer(OutBufLen);
    uint8_t* OutputReference = BufferOutputReference.GetBuffer(OutBufLen);

    std::default_random_engine generator(static_cast<unsigned>(N));

    std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
    float MinimumValue = min_gen(generator);

    std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
    float MaximumValue = max_gen(generator);

    float Scale = (MaximumValue - MinimumValue) / 32.f;

    std::uniform_int_distribution<int32_t> zp_distribution(MinVal(), MaxVal());
    int8_t ZeroPoint = static_cast<int8_t>(zp_distribution(generator));

    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);
    for (size_t n = 0; n < N; n++) {
      Input[n] = distribution(generator);
    }

    GenerateReference(Input, OutputReference, N, Scale, ZeroPoint);

    if constexpr (Signed) {
      MlasQuantizeLinearS4(Input, Output, N, Scale, ZeroPoint);
    } else {
      MlasQuantizeLinearU4(Input, Output, N, Scale, ZeroPoint);
    }

    for (size_t n = 0; n < N; n++) {
      size_t i = n >> 1;
      size_t j = n & 0x1;
      const uint8_t Shift = 4 * static_cast<uint8_t>(j);

      int32_t actual_val = (Output[i] >> Shift) & 0xF;
      int32_t expected_val = (OutputReference[i] >> Shift) & 0xF;

      if constexpr (Signed) {
        constexpr uint8_t SignExtShift = (sizeof(int32_t) * 8) - 4;
        actual_val = (actual_val << SignExtShift) >> SignExtShift;
        expected_val = (expected_val << SignExtShift) >> SignExtShift;
      }

      ASSERT_EQ(actual_val, expected_val) << ", size=" << N
                                          << ", index=" << n
                                          << ", nibble=" << j;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    if constexpr (Signed) {
      return "QuantizeLinearS4";
    } else {
      return "QuantizeLinearU4";
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
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<int8_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<uint8_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<int16_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinearTest<uint16_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinear4BitTest<false>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinear4BitTest<true>>::RegisterShortExecute();
  }
  return count;
});
