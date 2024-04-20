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

template <typename UnpackedType>
class MlasQuantizeLinear4BitTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<UnpackedType> BufferOutput;
  MatrixGuardBuffer<UnpackedType> BufferOutputReference;

  UnpackedType MinVal() const {
    if constexpr (std::is_same_v<UnpackedType, int8_t>) {
      return -8;
    } else if (std::is_same_v<UnpackedType, uint8_t>) {
      return 0;
    }
  }

  UnpackedType MaxVal() const {
    if constexpr (std::is_same_v<UnpackedType, int8_t>) {
      return 7;
    } else {
      static_assert(std::is_same_v<UnpackedType, uint8_t>);
      return 15;
    }
  }

  void GenerateReference(const float* Input, UnpackedType* OutputReference, size_t N, float Scale,
                         UnpackedType ZeroPoint) {
    for (size_t n = 0; n < N; n++) {
      float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
      FloatValue = std::max(FloatValue, static_cast<float>(MinVal()));
      FloatValue = std::min(FloatValue, static_cast<float>(MaxVal()));

      size_t i = n >> 1;
      size_t j = n & 0x1;

      UnpackedType IntValue = static_cast<UnpackedType>(FloatValue);

      if (j == 0) {
        OutputReference[i] = IntValue & 0xF;
      } else {
        OutputReference[i] |= static_cast<UnpackedType>((IntValue & 0xF) << 4);
      }
    }
  }

  void Test(size_t N) {
    size_t OutBufLen = (N + 1) / 2;
    float* Input = BufferInput.GetBuffer(N);
    UnpackedType* Output = BufferOutput.GetBuffer(OutBufLen);
    UnpackedType* OutputReference = BufferOutputReference.GetBuffer(OutBufLen);

    std::default_random_engine generator(static_cast<unsigned>(N));

    std::uniform_real_distribution<float> min_gen(-10.f, -10e-3f);
    float MinimumValue = min_gen(generator);

    std::uniform_real_distribution<float> max_gen(10e-3f, 10.f);
    float MaximumValue = max_gen(generator);

    float Scale = (MaximumValue - MinimumValue) / 32.f;

    std::uniform_int_distribution<int32_t> zp_distribution(MinVal(), MaxVal());
    UnpackedType ZeroPoint = static_cast<UnpackedType>(zp_distribution(generator));

    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);
    for (size_t n = 0; n < N; n++) {
      Input[n] = distribution(generator);
    }

    GenerateReference(Input, OutputReference, N, Scale, ZeroPoint);

    if constexpr (std::is_same_v<UnpackedType, int8_t>) {
      MlasQuantizeLinearS4(Input, Output, N, Scale, ZeroPoint);
    } else {
      static_assert(std::is_same_v<UnpackedType, uint8_t>);
      MlasQuantizeLinearU4(Input, Output, N, Scale, ZeroPoint);
    }

    for (size_t n = 0; n < N; n++) {
      size_t i = n >> 1;
      size_t j = n & 0x1;

      if (j == 0) {
        ASSERT_EQ(Output[i] & 0xF, OutputReference[i] & 0xF) << ", size=" << N
                                                             << ", index=" << n
                                                             << ", nibble=" << j;
      } else {
        ASSERT_EQ((Output[i] >> 4) & 0xF, (OutputReference[i] >> 4) & 0xF) << ", size=" << N
                                                                           << ", index=" << n
                                                                           << ", nibble=" << j;
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    if constexpr (std::is_same_v<UnpackedType, int8_t>) {
      return "QuantizeLinearS4";
    } else {
      static_assert(std::is_same_v<UnpackedType, uint8_t>);
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
    count += MlasDirectShortExecuteTests<MlasQuantizeLinear4BitTest<uint8_t>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasQuantizeLinear4BitTest<int8_t>>::RegisterShortExecute();
  }
  return count;
});
