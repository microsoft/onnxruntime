/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_cast_fp16.cpp

Abstract:

    Tests for MLAS FP16<->FP32 cast kernels.
    Verifies bit-exactness against MLAS_Half2Float / MLAS_Float2Half.

--*/

#include "test_util.h"
#include "mlas.h"
#include "mlas_float16.h"

#include <vector>

class MlasCastFp16Test : public MlasTestBase {
 public:
  void TestF16ToF32(size_t count) {
    std::vector<_mlas_fp16_> input(count);
    std::vector<float> output_ref(count);
    std::vector<float> output_dispatch(count);

    for (size_t i = 0; i < count; i++) {
      float val = (static_cast<float>(i % 2048) / 1024.0f) - 1.0f;
      input[i] = MLAS_Float2Half(val);
      output_ref[i] = MLAS_Half2Float(input[i]);
    }

    MlasConvertHalfToFloatBuffer(
        reinterpret_cast<const MLAS_FP16*>(input.data()),
        output_dispatch.data(), count);

    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(output_dispatch[i], output_ref[i])
          << "F16->F32 mismatch at [" << i << "], count=" << count;
    }
  }

  void TestF32ToF16(size_t count) {
    std::vector<float> input(count);
    std::vector<_mlas_fp16_> output_ref(count);
    std::vector<_mlas_fp16_> output_dispatch(count);

    for (size_t i = 0; i < count; i++) {
      input[i] = (static_cast<float>(i % 2048) / 1024.0f) - 1.0f;
      output_ref[i] = MLAS_Float2Half(input[i]);
    }

    MlasConvertFloatToHalfBuffer(
        input.data(),
        reinterpret_cast<MLAS_FP16*>(output_dispatch.data()), count);

    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(output_dispatch[i], output_ref[i])
          << "F32->F16 mismatch at [" << i << "], count=" << count;
    }
  }
};

class CastFp16ShortExecuteTest : public MlasTestFixture<MlasCastFp16Test> {
 public:
  CastFp16ShortExecuteTest(size_t count, bool f16_to_f32)
      : count_(count), f16_to_f32_(f16_to_f32) {}

  void TestBody() override {
    if (f16_to_f32_) {
      MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestF16ToF32(count_);
    } else {
      MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestF32ToF16(count_);
    }
  }

  static size_t RegisterSingleTest(size_t count, bool f16_to_f32) {
    std::stringstream ss;
    ss << "/" << (f16_to_f32 ? "F16toF32" : "F32toF16")
       << "/count" << count;
    auto test_name = ss.str();

    testing::RegisterTest(
        "CastFp16",
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasCastFp16Test>* {
          return new CastFp16ShortExecuteTest(count, f16_to_f32);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t cnt = 0;
    for (size_t n : {1, 7, 15, 16, 31, 32, 63, 64, 128, 255, 256, 1024, 65536}) {
      cnt += RegisterSingleTest(n, true);
      cnt += RegisterSingleTest(n, false);
    }
    return cnt;
  }

 private:
  size_t count_;
  bool f16_to_f32_;
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return CastFp16ShortExecuteTest::RegisterShortExecuteTests();
      }
      return 0;
    });
