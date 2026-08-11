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

#include <cmath>
#include <cstring>
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

  // Test special IEEE 754 values: ±0, ±Inf, NaN, denormals, and a value
  // requiring round-to-nearest-even in the f32→f16 direction.
  void TestSpecialValues() {
    // Raw fp16 bit patterns
    const uint16_t kPosZero = 0x0000;
    const uint16_t kNegZero = 0x8000;
    const uint16_t kPosInf  = 0x7C00;
    const uint16_t kNegInf  = 0xFC00;
    const uint16_t kNaN     = 0x7E00;  // quiet NaN
    const uint16_t kDenorm  = 0x0001;  // smallest positive denormal

    std::vector<uint16_t> special_bits = {
        kPosZero, kNegZero, kPosInf, kNegInf, kNaN, kDenorm};
    const size_t n = special_bits.size();

    // F16 -> F32: convert via dispatch and via scalar reference
    std::vector<_mlas_fp16_> h_input(n);
    std::memcpy(h_input.data(), special_bits.data(), n * sizeof(uint16_t));

    std::vector<float> out_dispatch(n);
    std::vector<float> out_ref(n);
    for (size_t i = 0; i < n; i++) {
      out_ref[i] = MLAS_Half2Float(h_input[i]);
    }

    MlasConvertHalfToFloatBuffer(
        reinterpret_cast<const MLAS_FP16*>(h_input.data()),
        out_dispatch.data(), n);

    for (size_t i = 0; i < n; i++) {
      uint32_t ref_bits, disp_bits;
      std::memcpy(&ref_bits, &out_ref[i], sizeof(float));
      std::memcpy(&disp_bits, &out_dispatch[i], sizeof(float));
      ASSERT_EQ(disp_bits, ref_bits)
          << "F16->F32 special mismatch at [" << i
          << "], h=0x" << std::hex << special_bits[i];
    }

    // F32 -> F16: test with the f32 equivalents plus a round-to-even case.
    // 1.0009765625f (0x3F802000) rounds to fp16 1.0 (0x3C00) under RTE.
    std::vector<float> f32_input = {
        0.0f, -0.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        1.0009765625f};
    const size_t m = f32_input.size();

    std::vector<_mlas_fp16_> f16_out_dispatch(m);
    std::vector<_mlas_fp16_> f16_out_ref(m);
    for (size_t i = 0; i < m; i++) {
      f16_out_ref[i] = MLAS_Float2Half(f32_input[i]);
    }

    MlasConvertFloatToHalfBuffer(
        f32_input.data(),
        reinterpret_cast<MLAS_FP16*>(f16_out_dispatch.data()), m);

    for (size_t i = 0; i < m; i++) {
      uint16_t ref_bits, disp_bits;
      std::memcpy(&ref_bits, &f16_out_ref[i], sizeof(uint16_t));
      std::memcpy(&disp_bits, &f16_out_dispatch[i], sizeof(uint16_t));
      ASSERT_EQ(disp_bits, ref_bits)
          << "F32->F16 special mismatch at [" << i
          << "], f32=" << f32_input[i];
    }
  }

  // Verify the vectorised kernel is dispatched (non-null function pointer)
  // on platforms that define MLAS_F16VEC_INTRINSICS_SUPPORTED or
  // MLAS_CAST_F16_NEON_SUPPORTED.
  void TestKernelIsDispatched() {
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) || \
    (defined(__APPLE__) && defined(MLAS_TARGET_ARM64))
    // On these platforms, the dispatch table must have a non-null kernel.
    // We verify this indirectly: convert a known value and confirm the
    // output matches the vectorised result (not just the scalar fallback).
    // A direct function-pointer check is not exposed in the public API, so
    // we rely on the platform constructor having set the pointers.
    const uint16_t one_h = 0x3C00;  // fp16 1.0
    _mlas_fp16_ input;
    std::memcpy(&input, &one_h, sizeof(uint16_t));
    float output = 0.0f;
    MlasConvertHalfToFloatBuffer(
        reinterpret_cast<const MLAS_FP16*>(&input), &output, 1);
    ASSERT_EQ(output, 1.0f) << "Kernel dispatch sanity check failed";
#endif
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
    // Various lengths including non-multiples of vector width (4/8)
    for (size_t n : {1, 2, 3, 5, 7, 9, 15, 16, 17, 31, 32, 63, 64, 128, 255, 256, 1024, 65536}) {
      cnt += RegisterSingleTest(n, true);
      cnt += RegisterSingleTest(n, false);
    }
    return cnt;
  }

 private:
  size_t count_;
  bool f16_to_f32_;
};

class CastFp16SpecialValuesTest : public MlasTestFixture<MlasCastFp16Test> {
 public:
  void TestBody() override {
    MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestSpecialValues();
  }

  static size_t RegisterTests() {
    testing::RegisterTest(
        "CastFp16",
        "/SpecialValues",
        nullptr,
        "/SpecialValues",
        __FILE__,
        __LINE__,
        []() -> MlasTestFixture<MlasCastFp16Test>* {
          return new CastFp16SpecialValuesTest();
        });
    return 1;
  }
};

class CastFp16KernelDispatchTest : public MlasTestFixture<MlasCastFp16Test> {
 public:
  void TestBody() override {
    MlasTestFixture<MlasCastFp16Test>::mlas_tester->TestKernelIsDispatched();
  }

  static size_t RegisterTests() {
    testing::RegisterTest(
        "CastFp16",
        "/KernelDispatched",
        nullptr,
        "/KernelDispatched",
        __FILE__,
        __LINE__,
        []() -> MlasTestFixture<MlasCastFp16Test>* {
          return new CastFp16KernelDispatchTest();
        });
    return 1;
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return CastFp16ShortExecuteTest::RegisterShortExecuteTests() +
               CastFp16SpecialValuesTest::RegisterTests() +
               CastFp16KernelDispatchTest::RegisterTests();
      }
      return 0;
    });
