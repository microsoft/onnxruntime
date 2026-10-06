/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm_fp16_c_fp32path.cpp

Abstract:

    Tests for the fp16 direct C output of the 4-bit CompFp32 MatMulNBits path.
    The fp16 result must be bit-identical to computing the fp32 result and
    converting it with MlasConvertFloatToHalfBuffer.

--*/

#ifndef ORT_MINIMAL_BUILD

#include "test_util.h"
#include "mlas_qnbit.h"
#include "mlas_q4.h"
#include "core/common/float16.h"

class MlasSQNBitGemmFp16CFp32PathTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<uint8_t> BufferQuantBData;
  MatrixGuardBuffer<float> BufferQuantBScale;
  MatrixGuardBuffer<uint8_t> BufferQuantBZeroPoint;
  MatrixGuardBuffer<std::byte> BufferPackedQuantBData;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferCFp32;
  MatrixGuardBuffer<MLAS_FP16> BufferCFp16;
  MatrixGuardBuffer<MLAS_FP16> BufferCFp16Ref;

  void RunOne(size_t M, size_t N, size_t K, size_t BlkLen,
              bool Symmetric, bool WithBias, bool WithThreadpool) {
    constexpr size_t BlkBitWidth = 4;

    if (!MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, SQNBIT_CompFp32) ||
        !MlasQNBitGemmFp16DirectCOutputSupported(BlkBitWidth, SQNBIT_CompFp32)) {
      GTEST_SKIP() << "fp16 direct C for CompFp32 is not available on this platform.";
    }

    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    float* A = BufferA.GetBuffer(M * K);
    float* B = BufferB.GetBuffer(K * N);
    for (size_t i = 0; i < M * K; ++i) {
      A[i] = static_cast<float>(static_cast<int>((i * 7919u + 13u) % 2001u) - 1000) / 967.0f;
    }
    for (size_t i = 0; i < K * N; ++i) {
      B[i] = static_cast<float>(static_cast<int>((i * 104729u + 31u) % 2001u) - 1000) / 1013.0f;
    }

    size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
    MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(static_cast<int>(BlkLen), /* columnwise */ true,
                                                   static_cast<int>(K), static_cast<int>(N),
                                                   QuantBDataSizeInBytes, QuantBScaleSize,
                                                   &QuantBZeroPointSizeInBytes);

    uint8_t* QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
    float* QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);
    uint8_t* QuantBZeroPoint = Symmetric ? nullptr : BufferQuantBZeroPoint.GetBuffer(QuantBZeroPointSizeInBytes);

    MlasQuantizeBlockwise<float, BlkBitWidth>(QuantBData, QuantBScale, QuantBZeroPoint, B,
                                              static_cast<int>(BlkLen), /* columnwise */ true,
                                              static_cast<int>(K), static_cast<int>(N),
                                              static_cast<int>(N), GetMlasThreadPool());

    void* PackedQuantBData = nullptr;
    if (const auto PackedQuantBDataSize = MlasQNBitGemmPackQuantBDataSize(
            N, K, BlkBitWidth, BlkLen, !Symmetric, SQNBIT_CompFp32, nullptr);
        PackedQuantBDataSize > 0) {
      PackedQuantBData = BufferPackedQuantBData.GetBuffer(PackedQuantBDataSize);
      MlasQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, SQNBIT_CompFp32, QuantBData,
                                  PackedQuantBData, QuantBScale, !Symmetric, QuantBZeroPoint,
                                  GetMlasThreadPool(), nullptr);
    }

    float* Bias = WithBias ? BufferBias.GetBuffer(N) : nullptr;
    if (Bias != nullptr) {
      for (size_t n = 0; n < N; ++n) {
        Bias[n] = static_cast<float>(static_cast<int>((n * 379u + 3u) % 201u) - 100) / 100.0f;
      }
    }

    float* CFp32 = BufferCFp32.GetBuffer(M * N, true);
    MLAS_FP16* CFp16 = BufferCFp16.GetBuffer(M * N, true);
    MLAS_FP16* CFp16Ref = BufferCFp16Ref.GetBuffer(M * N, true);

    auto call_gemm = [&](float* c, MLAS_FP16* c_fp16) {
      MLAS_QNBIT_GEMM_DATA_PARAMS<float> params;
      params.A = A;
      params.lda = K;
      params.QuantBDataWorkspace = QuantBData;
      params.PackedQuantBData = static_cast<const std::byte*>(
          PackedQuantBData != nullptr ? PackedQuantBData : static_cast<const void*>(QuantBData));
      params.QuantBScale = QuantBScale;
      params.QuantBZeroPoint = QuantBZeroPoint;
      params.Bias = Bias;
      params.C = c;
      params.CFp16 = c_fp16;
      params.ldc = N;
      MlasQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, SQNBIT_CompFp32, &params,
                         nullptr, Threadpool, nullptr);
    };

    // fp32 reference pass, then the fp16 direct pass over the same inputs.
    call_gemm(CFp32, nullptr);
    call_gemm(nullptr, CFp16);

    MlasConvertFloatToHalfBuffer(CFp32, CFp16Ref, M * N);

    for (size_t i = 0; i < M * N; ++i) {
      ASSERT_EQ(CFp16[i].val, CFp16Ref[i].val)
          << "mismatch @" << i << " M=" << M << " N=" << N << " K=" << K
          << " BlkLen=" << BlkLen << " sym=" << Symmetric << " bias=" << WithBias
          << " tp=" << WithThreadpool;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "SQNBitGemmFp16CFp32Path";
  }

  void ExecuteShort(void) override {
    for (size_t BlkLen : {16, 32, 64, 128}) {
      // M == 1 takes the M1 kernel, M > 1 the dequant-strip SGEMM loop; N values
      // cover the 128 and 32 column chunkings and their remainders.
      RunOne(1, 128, 256, BlkLen, false, false, false);
      RunOne(1, 130, 129, BlkLen, true, true, false);
      RunOne(4, 32, 128, BlkLen, false, true, false);
      RunOne(7, 129, 257, BlkLen, true, false, false);
      RunOne(33, 63, 512, BlkLen, false, true, true);
      RunOne(129, 257, 1031, BlkLen, true, true, true);
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasSQNBitGemmFp16CFp32PathTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // ORT_MINIMAL_BUILD
