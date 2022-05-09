// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

namespace{
    inline uint32_t BitsOfFp32(float FloatValue) {
  union {
    uint32_t IntegerValue;
    float FloatValue;
  } u;
  u.FloatValue = FloatValue;
  return u.IntegerValue;
}

    inline void ComputeMultiplierShift(float scale, int32_t& multiplier, int32_t& pre_shift, int32_t& post_shift) {
  // Compute requantization parameters.
  const uint32_t scale_bits = BitsOfFp32(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);

  // Shift is in [-8, 31] range.
  const int32_t shift = 127 + 31 - 32 - (scale_bits >> 23);
  assert(shift >= -8);
  assert(shift < 32);

  // Split shift into pre_shift + post_shift, post_shift in [1, 31] range.
  post_shift = shift > 1 ? shift : 1;
  pre_shift = shift - post_shift;

  pre_shift = -pre_shift;
  post_shift = -post_shift;
}

inline void ComputeMultiplierShiftVector(
    const std::vector<float>& scales,
    std::vector<int32_t>& multipliers,
    std::vector<int32_t>& pre_shifts,
    std::vector<int32_t>& post_shifts) {
  multipliers.resize(scales.size());
  pre_shifts.resize(scales.size());
  post_shifts.resize(scales.size());

  for (size_t i = 0; i < scales.size(); i++) {
    ComputeMultiplierShift(scales[i], multipliers[i], pre_shifts[i], post_shifts[i]);
  }
}

template<typename T>
std::string TypeName();

template<>
std::string TypeName<int8_t>(){
  return "int8";
}

template<>
std::string TypeName<uint8_t>(){
  return "uint8";
}

}

template <typename ElementType>
class MlasMlasRequantizeOutputTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<int32_t> BufferInput;
  MatrixGuardBuffer<ElementType> BufferOutput;
  MatrixGuardBuffer<ElementType> BufferOutputReference;

  void
  Test(size_t M, size_t N) {
    int32_t* Input = BufferInput.GetBuffer(M * N);
    ElementType* Output = BufferOutput.GetBuffer(M * N);
    ElementType* OutputReference = BufferOutputReference.GetBuffer(M * N);

    for(size_t i = 0; i < M*N; i++) {
      //Input[i] = rand() % 600000;
      Input[i] = 547793;
    }

    float scale = 0.000543;
    int32_t multiplier, pre_shift, post_shift;
    ComputeMultiplierShift(scale, multiplier, pre_shift, post_shift);
    MLAS_REQUANT_PARAM RequantParam(&multiplier,
                       &pre_shift,
                       &post_shift,
                       1,
                       0);
    MlasRequantizeOutput(Input, N, Output, N, nullptr, &RequantParam, 0, 0, M, N);

    ReferenceRequantizeOutput(Input, OutputReference, M, N, scale);

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        if(Output[m * M + n] != OutputReference[m * M + n]){
          std::cout<<"mismatch:"<<m<<","<<n<<std::endl;
          ASSERT_EQ(Output[m * M + n], OutputReference[m * M + n]);
        }
      }
    }
    ASSERT_EQ(memcmp(Output, OutputReference, M * N * sizeof(ElementType)), 0) << " [" << M << "," << N << "]";
  }

  void ReferenceRequantizeOutput(const int32_t* Input, ElementType* Output, size_t M, size_t N, float scale) {
    std::cout<<"M:"<<M<<",N:"<<N<<":";
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
          float value = Input[m * N + n] * scale;
          value = std::nearbyint(value);
          float max_value = std::numeric_limits<ElementType>::max();
          float min_value = std::numeric_limits<ElementType>::min();
          value = std::max(std::min(max_value, value), min_value);
          Output[m * N + n] = static_cast<ElementType>(value);
          std::cout<<"["<<Input[m * N + n]<<","<<(int32_t)Output[m * N + n]<<"],";
      }
      std::cout<<std::endl;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name = std::string("Requantize_") + TypeName<ElementType>();
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t m = 1; m <= 1; m++) {
      for (size_t n = 1; n <= 43; n++) {
        Test(m, n);
      }
    }
  }
};

template <> MlasMlasRequantizeOutputTest<int8_t>* MlasTestFixture<MlasMlasRequantizeOutputTest<int8_t>>::mlas_tester(nullptr);
template <> MlasMlasRequantizeOutputTest<uint8_t>* MlasTestFixture<MlasMlasRequantizeOutputTest<uint8_t>>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
      count += MlasDirectShortExecuteTests<MlasMlasRequantizeOutputTest<int8_t>>::RegisterShortExecute();
      count += MlasDirectShortExecuteTests<MlasMlasRequantizeOutputTest<uint8_t>>::RegisterShortExecute();
  }
  return count;
});
