// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "mlas_q4.h"

void MlasSgemmCopyPackB(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountX,
    size_t CountY);

class MlasQ4dqTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> FpInputBuf;
  MatrixGuardBuffer<uint8_t> PackedBuf;
  MatrixGuardBuffer<float> FpOutBuf;
  MatrixGuardBuffer<float> SgemmPackBuf;
  MatrixGuardBuffer<float> SgemmPackRefBuf;

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

    /* Test MlasBlkQ4DequantSgemmPackB, make sure we can reuse SGEMM kernel as it rearrange B the same way as sgemm pack B*/
    const size_t AlignedN = (N + 15) & ~15;
    const size_t AlignedK = (K + 15) & ~15;
    float* gemmpack = SgemmPackBuf.GetBuffer(AlignedK * AlignedN, true);
    float* gemmpack_ref = SgemmPackRefBuf.GetBuffer(AlignedK * AlignedN, true);
    MlasSgemmCopyPackB(gemmpack_ref, Input, N, N, K);

    const size_t blkq_ldb = MlasQ4GemmPackBSize(qtype, 1, K);
    MlasBlkQ4DequantSgemmPackB(qtype, gemmpack, Packed, N, K, blkq_ldb);
    for (size_t i = 0; i < AlignedN * K; i++) {
      ASSERT_EQ(gemmpack[i], gemmpack_ref[i]) << ", sgemm pack index=" << i << ", [" << N << "x"
                                     << K << "] QType: " << qtype;
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
