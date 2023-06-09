// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "mlas_q4.h"

class MlasQ4dqTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> FpInputBuf;
  MatrixGuardBuffer<uint8_t> PackedBuf;
  MatrixGuardBuffer<float> FpOutBuf;

  void Test(size_t N, size_t K, MLAS_BLK_QUANT_TYPE qtype) {
    float* Input = FpInputBuf.GetBuffer(N * K, true);
    if (qtype != BlkQ4Zp8) {
      int v = -7;
      for (size_t i = 0; i < N * K; i++) {
        if (v == 0 || v == -3 || v == 3) {
          v++;
        }
        Input[i] = (float)v;
        if (++v >= 8) {
          v = -8;
        }
      }
    } else {
      int v = 0;
      for (size_t i = 0; i < N * K; i++) {
        Input[i] = (float)v;
        if (++v >= 16) {
          v = -0;
        }
      }
    }

    size_t qsize = MlasQ4GemmPackBSize(qtype, N, K);
    uint8_t* Packed = PackedBuf.GetBuffer(qsize, true);
    float* Output = FpOutBuf.GetBuffer(N * K, true);

    MlasQ4GemmPackB(qtype, Packed, Input, N, K, N);
    MlasQ4GemmUnPackB(qtype, Output, Packed, N, K, N);

    for (size_t i = 0; i < N * K; i++) {
      ASSERT_EQ(Output[i], Input[i]) << ", index=" << i << ", [" << N << "x"
                                     << K << "] QType: " << qtype ;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Q4DQ");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    Test(1, 20, BlkQ4Sym);
    Test(1, 20, BlkQ4Zp8);
    Test(1, 52, BlkQ4Sym);
    Test(1, 52, BlkQ4Zp8);
    Test(1, 52, BlkQ4Sym64);
    Test(3, 20, BlkQ4Sym);
    Test(3, 20, BlkQ4Zp8);
    Test(3, 52, BlkQ4Sym);
    Test(3, 52, BlkQ4Zp8);
    Test(3, 52, BlkQ4Sym64);
    Test(static_cast<size_t>(4 * 10) + 1, static_cast<size_t>(32 * 9) + 17, BlkQ4Zp8);
    Test(static_cast<size_t>(4 * 10) + 1, static_cast<size_t>(32 * 9) + 17, BlkQ4Sym);
    Test(static_cast<size_t>(4 * 10) + 1, static_cast<size_t>(32 * 9) + 17, BlkQ4Sym64);
    Test(static_cast<size_t>(4 * 10) + 1, static_cast<size_t>(32 * 9) + 17, BlkQ4Sym128);
    Test(static_cast<size_t>(4 * 20) + 3, static_cast<size_t>(32 * 15) + 17, BlkQ4Zp8);
    Test(static_cast<size_t>(4 * 20) + 3, static_cast<size_t>(32 * 15) + 17, BlkQ4Sym);
    Test(static_cast<size_t>(4 * 20) + 3, static_cast<size_t>(32 * 15) + 17, BlkQ4Sym64);
    Test(static_cast<size_t>(4 * 20) + 3, static_cast<size_t>(32 * 15) + 17, BlkQ4Sym128);
  }

  MlasQ4dqTest() = default;
};

template <>
MlasQ4dqTest* MlasTestFixture<MlasQ4dqTest>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasQ4dqTest>::RegisterShortExecute();
  }
  return count;
});
