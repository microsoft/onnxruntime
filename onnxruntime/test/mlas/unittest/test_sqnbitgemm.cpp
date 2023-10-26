/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm.h

Abstract:

    Tests for MLAS n-bit int block quantized GEMM.

--*/

#pragma once

#include "test_util.h"
#include "mlas_qnbit.h"

namespace {

constexpr size_t DivRoundUp(size_t a, size_t b) {
  return (a + b - 1) / b;
}

constexpr size_t BlkDataSizeInBytes(size_t BlkBitWidth, size_t BlkLen) {
  return BlkLen * BlkBitWidth / 8;
}

template <size_t BlkBitWidth>
constexpr size_t ZeroPointsForBlksSizeInBytes(size_t BlkCount) {
  if constexpr (BlkBitWidth <= 4) {
    return DivRoundUp(BlkCount, 2);
  } else {
    return BlkCount;
  }
}

template <size_t BlkLen, size_t BlkBitWidth>
struct ReferenceQNBitPacking {
  static_assert(BlkBitWidth == 4, "Only implemented for BlkBitWidth == 4.");

  static void GetPackedBSizes(size_t CountN, size_t CountK,
                              size_t& PackedBDataSizeInBytes,
                              size_t& PackedBScaleElementCount,
                              size_t* PackedBZeroPointSizeInBytes) {
    const size_t BlockCountK = DivRoundUp(CountK, BlkLen);
    const size_t TotalBlockCount = CountN * BlockCountK;

    PackedBDataSizeInBytes = TotalBlockCount * BlkDataSizeInBytes(BlkLen, BlkBitWidth);
    PackedBScaleElementCount = TotalBlockCount;
    if (PackedBZeroPointSizeInBytes) {
      *PackedBZeroPointSizeInBytes = CountN * ZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);
    }
  }

  static void PackB(size_t CountN, size_t CountK,
                    const float* BDataPtr, size_t ldb,
                    uint8_t* PackedBDataPtr,
                    float* PackedBScalePtr,
                    uint8_t* PackedBZeroPointPtr) {
    const size_t BlockCountK = DivRoundUp(CountK, BlkLen);

    uint8_t* PackedBDataColPtr = PackedBDataPtr;
    float* PackedBScaleColPtr = PackedBScalePtr;
    uint8_t* PackedBZeroPointColPtr = PackedBZeroPointPtr;

    for (size_t n = 0; n < CountN; ++n) {
      for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
        size_t kklen = std::min(BlkLen, CountK - k);

        uint8_t* PackedBBlkDataPtr = PackedBDataColPtr + k_blk_idx * BlkDataSizeInBytes(BlkLen, BlkBitWidth);

        if (PackedBZeroPointColPtr) {
          float scale_block;
          uint8_t zp_block;
          QuantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block, zp_block);

          if ((k_blk_idx & 1) == 0) {
            PackedBZeroPointColPtr[k_blk_idx / 2] = zp_block & 0x0F;
          } else {
            PackedBZeroPointColPtr[k_blk_idx / 2] |= zp_block << 4;
          }

          PackedBScaleColPtr[k_blk_idx] = scale_block;
        } else {
          float scale_block;
          QuantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block);

          PackedBScaleColPtr[k_blk_idx] = scale_block;
        }
      }

      PackedBDataColPtr += BlockCountK * BlkDataSizeInBytes(BlkLen, BlkBitWidth);
      PackedBScaleColPtr += BlockCountK;
      if (PackedBZeroPointColPtr != nullptr) {
        PackedBZeroPointColPtr += ZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);
      }
    }
  }

  static void UnpackB(size_t CountN, size_t CountK,
                      const uint8_t* PackedBDataPtr, const float* PackedBScalePtr, const uint8_t* PackedBZeroPointPtr,
                      float* BDataPtr, size_t ldb) {
    const size_t BlockCountK = DivRoundUp(CountK, BlkLen);

    const uint8_t* PackedBDataColPtr = PackedBDataPtr;
    const float* PackedBScaleColPtr = PackedBScalePtr;
    const uint8_t* PackedBZeroPointColPtr = PackedBZeroPointPtr;

    for (size_t n = 0; n < CountN; ++n) {
      for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
        size_t kklen = std::min(BlkLen, CountK - k);

        const uint8_t* PackedBBlkDataPtr = PackedBDataColPtr + k_blk_idx * BlkDataSizeInBytes(BlkLen, BlkBitWidth);
        const float scale_block = PackedBScaleColPtr[k_blk_idx];

        if (PackedBZeroPointColPtr) {
          const uint8_t zp_block = ((k_blk_idx & 1) == 1)
                                       ? (PackedBZeroPointColPtr[k_blk_idx / 2] >> 4)
                                       : (PackedBZeroPointColPtr[k_blk_idx / 2] & 0x0F);

          DequantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block, zp_block);
        } else {
          DequantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block);
        }
      }

      PackedBDataColPtr += BlockCountK * BlkDataSizeInBytes(BlkLen, BlkBitWidth);
      PackedBScaleColPtr += BlockCountK;
      if (PackedBZeroPointColPtr != nullptr) {
        PackedBZeroPointColPtr += ZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);
      }
    }
  }

  static void QuantizeBlock(const float* b_begin, size_t ldb, size_t actual_block_size,
                            uint8_t* data_block, float& scale_block, uint8_t& zp_block) {
    float min = *b_begin;
    float max = *b_begin;
    for (int32_t kk = 0; kk < actual_block_size; kk++) {
      const float v = b_begin[ldb * kk];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    min = std::min(min, 0.0f);
    max = std::max(max, 0.0f);

    scale_block = (max - min) / ((1 << 4) - 1);

    const float reciprocal_scale = scale_block ? 1.0f / scale_block : 0.0f;
    float zero_point_fp = min;
    if (scale_block != 0.0f) {
      zero_point_fp = 0.f - min / scale_block;
    }

    // Handle any clamping
    if (zero_point_fp < 0.0f) {
      zp_block = 0;
    } else if (zero_point_fp > 15.0f) {
      zp_block = 15;
    } else {
      zp_block = (uint8_t)roundf(zero_point_fp);
    }

    for (int32_t kk = 0; kk < actual_block_size; kk += 2) {
      const float v0 = b_begin[ldb * kk];
      const uint8_t vi0 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v0 * reciprocal_scale + zp_block)));

      const float v1 = (kk + 1 < actual_block_size) ? b_begin[ldb * (kk + 1)] : 0.f;
      const uint8_t vi1 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v1 * reciprocal_scale + zp_block)));

      data_block[kk / 2] = vi0 | (vi1 << 4);
    }
  }

  static void QuantizeBlock(const float* b_begin, size_t ldb, size_t actual_block_size,
                            uint8_t* data_block, float& scale_block) {
    float amax = 0.0f;  // abs(max)
    float max = 0.0f;

    for (int32_t kk = 0; kk < actual_block_size; kk++) {
      const float v = b_begin[ldb * kk];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    scale_block = max / (-8.f);
    const float reciprocal_scale = scale_block ? 1.0f / scale_block : 0.0f;

    for (int32_t kk = 0; kk < actual_block_size; kk += 2) {
      const float v0 = b_begin[ldb * kk] * reciprocal_scale;
      const uint8_t vi0 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v0 + 8.f)));

      const float v1 = (kk + 1 < actual_block_size) ? b_begin[ldb * (kk + 1)] * reciprocal_scale : 0;
      const uint8_t vi1 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v1 + 8.f)));

      data_block[kk / 2] = vi0 | (vi1 << 4);
    }
  }

  static void DequantizeBlock(float* b_begin, size_t ldb, size_t actual_block_size,
                              const uint8_t* data_block, float scale_block, uint8_t zp_block) {
    for (size_t kk = 0; kk < actual_block_size; kk += 2) {
      float x0 = static_cast<float>(data_block[kk / 2] & 0x0F);
      b_begin[ldb * kk] = scale_block * (x0 - zp_block);

      if (kk + 1 < actual_block_size) {
        float x1 = static_cast<float>(data_block[kk / 2] >> 4);
        b_begin[ldb * (kk + 1)] = scale_block * (x1 - zp_block);
      }
    }
  }

  static void DequantizeBlock(float* b_begin, size_t ldb, size_t actual_block_size,
                              const uint8_t* data_block, float scale_block) {
    DequantizeBlock(b_begin, ldb, actual_block_size, data_block, scale_block, uint8_t{8});
  }
};

}  // namespace

/**
 * @brief Test class for n-bit int block quantized GEMM
 *        Note: only 2-D matmul supported for now
 */
template <size_t BlkLen, size_t BlkBitWidth>
class MlasSQNBitGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<uint8_t> BufferPackedBData;
  MatrixGuardBuffer<uint8_t> BufferPackedBZeroPoint;
  MatrixGuardBuffer<float> BufferPackedBScale;
  MatrixGuardBuffer<float> BufferUnpackedBReference;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCReference;

  void CallGemm(size_t M,
                size_t N,
                size_t K,
                const float* A,
                size_t lda,
                const uint8_t* PackedBData,
                const float* PackedBScale,
                const uint8_t* PackedBZeroPoint,
                const float* Bias,
                float* C,
                size_t ldc,
                MLAS_THREADPOOL* Threadpool) {
    MLAS_SQNBIT_GEMM_DATA_PARAMS params;
    params.A = A;
    params.lda = lda;
    params.Bias = Bias;
    params.C = C;
    params.ldc = ldc;
    params.PackedBData = PackedBData;
    params.PackedBScale = PackedBScale;
    params.PackedBZeroPoint = PackedBZeroPoint;
    params.PostProcessor = nullptr;

    MlasSQNBitGemmBatch(M, N, K, 1, BlkLen, BlkBitWidth, &params, Threadpool);
  }

  void CallReferenceGemm(size_t M,
                         size_t N,
                         size_t K,
                         const float* A,
                         const uint8_t* PackedBData,
                         const float* PackedBScale,
                         const uint8_t* PackedBZeroPoint,
                         const float* Bias,
                         float* C) {
    float* UnpackedBData = BufferUnpackedBReference.GetBuffer(K * N);
    ReferenceQNBitPacking<BlkLen, BlkBitWidth>::UnpackB(
        N, K, PackedBData, PackedBScale, PackedBZeroPoint, UnpackedBData, N);

    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        const float* a = A + m * K;
        const float* b = UnpackedBData + n;
        float* c = C + (m * N) + n;

        float sum = Bias == nullptr ? 0.0f : Bias[n];
        for (size_t k = 0; k < K; k++) {
          sum += (*a) * (*b);
          b += N;
          a += 1;
        }
        *c = sum;
      }
    }
  }

 public:
  void Test(size_t M, size_t N, size_t K,
            bool WithBias, bool WithZeroPoint, bool WithThreadpool) {
    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    const float* A = BufferA.GetBuffer(K * M);

    const float* B = BufferB.GetBuffer(N * K);

    const float* Bias = nullptr;
    if (WithBias) {
      Bias = BufferBias.GetBuffer(N);
    }

#if 0
    auto print_matrix = [](size_t ncols, size_t nrows, const float* data) {
      for (size_t row = 0; row < nrows; ++row) {
        for (size_t col = 0; col < ncols; ++col) {
          std::cout << data[row * nrows + col] << "\t";
        }
        std::cout << "\n";
      }
    };

    std::cout << "A:\n";
    print_matrix(M, K, A);
    std::cout << "B:\n";
    print_matrix(K, N, B);
#endif

    float* C = BufferC.GetBuffer(N * M, true);
    float* CReference = BufferCReference.GetBuffer(N * M, true);

    // pack B
    uint8_t* PackedBData = nullptr;
    float* PackedBScale = nullptr;
    uint8_t* PackedBZeroPoint = nullptr;
    {
      size_t PackedBDataSize, PackedBScaleSize, PackedBZeroPointSize;
      ReferenceQNBitPacking<BlkLen, BlkBitWidth>::GetPackedBSizes(
          N, K, PackedBDataSize, PackedBScaleSize, &PackedBZeroPointSize);

      PackedBData = BufferPackedBData.GetBuffer(PackedBDataSize);
      PackedBScale = BufferPackedBScale.GetBuffer(PackedBScaleSize);
      if (WithZeroPoint) {
        PackedBZeroPoint = BufferPackedBZeroPoint.GetBuffer(PackedBZeroPointSize);
      }

      ReferenceQNBitPacking<BlkLen, BlkBitWidth>::PackB(N, K, B, /* ldb */ N,
                                                        PackedBData, PackedBScale, PackedBZeroPoint);
    }

    CallGemm(M, N, K, A, /* lda */ K, PackedBData, PackedBScale, PackedBZeroPoint, Bias, C, /* ldc */ N, Threadpool);
    CallReferenceGemm(M, N, K, A, PackedBData, PackedBScale, PackedBZeroPoint, Bias, CReference);

    size_t f = 0;
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++, f++) {
        ASSERT_TRUE(CloseEnough(C[f], CReference[f]))
            << "Expected: " << CReference[f] << " Actual: " << C[f] << "@[" << m << "x" << n << "], "
            << "M=" << M << ", N=" << N << ", K=" << K;
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("SQNBitGemm") +
                                    "BlkLen" + std::to_string(BlkLen) +
                                    "BlkBitWidth" + std::to_string(BlkBitWidth);
    return suite_name.c_str();
  }
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <size_t BlkLen, size_t BlkBitWidth>
class SQNBitGemmShortExecuteTest : public MlasTestFixture<MlasSQNBitGemmTest<BlkLen, BlkBitWidth>> {
 public:
  explicit SQNBitGemmShortExecuteTest(size_t M, size_t N, size_t K,
                                      bool WithThreadpool, bool WithZeroPoint, bool WithBias)
      : M_(M), N_(N), K_(K), WithThreadpool_(WithThreadpool), WithZeroPoint_(WithZeroPoint), WithBias_(WithBias) {
  }

  void TestBody() override {
    MlasTestFixture<MlasTesterType>::mlas_tester->Test(
        M_, N_, K_, WithThreadpool_, WithZeroPoint_, WithBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K,
                                   bool WithThreadpool, bool WithZeroPoint, bool WithBias) {
    std::stringstream ss;
    ss << (WithThreadpool ? "SingleThread" : "Threaded")
       << "/hasZeroPoint" << WithZeroPoint
       << "/M" << M << "xN" << N << "xK" << K
       << "/hasBias" << WithBias;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasTesterType::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasTesterType>* {
          return new SQNBitGemmShortExecuteTest(
              M, N, K, WithThreadpool, WithZeroPoint, WithBias);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (bool WithThreadpool : {false, true}) {
      for (bool WithZeroPoint : {false, true}) {
        for (size_t b = 1; b < 16; b++) {
          test_registered += RegisterSingleTest(b, b, b, WithThreadpool, WithZeroPoint, false);
          test_registered += RegisterSingleTest(b, b, b, WithThreadpool, WithZeroPoint, true);
        }
        for (size_t b = 16; b <= 256; b <<= 1) {
          test_registered += RegisterSingleTest(b, b, b, WithThreadpool, WithZeroPoint, false);
          test_registered += RegisterSingleTest(b, b, b, WithThreadpool, WithZeroPoint, true);
        }
        for (size_t b = 256; b < 320; b += 32) {
          test_registered += RegisterSingleTest(b, b, b, WithThreadpool, WithZeroPoint, true);
        }
        for (size_t b = 1; b < 96; b++) {
          test_registered += RegisterSingleTest(1, b, 32, WithThreadpool, WithZeroPoint, false);
          test_registered += RegisterSingleTest(1, 32, b, WithThreadpool, WithZeroPoint, true);
          test_registered += RegisterSingleTest(1, b, b, WithThreadpool, WithZeroPoint, false);
        }
        test_registered += RegisterSingleTest(43, 500, 401, WithThreadpool, WithZeroPoint, true);

        // test_registered += RegisterSingleTest(1001, 1027, 1031, WithThreadpool, WithZeroPoint, false);
      }
    }

    return test_registered;
  }

 private:
  size_t M_, N_, K_;
  bool WithThreadpool_, WithZeroPoint_, WithBias_;
};

template <>
MlasSQNBitGemmTest<32, 4>* MlasTestFixture<MlasSQNBitGemmTest<32, 4>>::mlas_tester(nullptr);

static size_t SQNBitGemmRegisterAllShortExecuteTests() {
  size_t count = 0;

  count += SQNBitGemmShortExecuteTest<32, 4>::RegisterShortExecuteTests();

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (is_short_execute) {
    return SQNBitGemmRegisterAllShortExecuteTests() > 0;
  }
  return false;
});
