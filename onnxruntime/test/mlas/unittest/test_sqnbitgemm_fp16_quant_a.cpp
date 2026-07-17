/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm_fp16_quant_a.cpp

Abstract:

    Checks that the fp16 direct-quantize-A path for the CompInt8 GEMM
    (MLAS_QNBIT_GEMM_DATA_PARAMS::AFp16) produces byte-identical output to
    converting A to fp32 first and running the float quantizer. Both runs are
    given the same activations (the fp16 values, and those same values widened to
    float), so any difference would be a bug in the fused fp16 quantizer.

--*/

#ifndef ORT_MINIMAL_BUILD

#include "test_util.h"
#include "mlas_qnbit.h"
#include "mlas_q4.h"
#include "core/common/float16.h"

#include <random>
#include <string>

using onnxruntime::MLFloat16;

template <size_t BlkLen>
class MlasQNBitGemmFp16QuantATest : public MlasTestBase {
  static constexpr size_t BlkBitWidth = 4;

  MatrixGuardBuffer<float> BufferAFloat;
  MatrixGuardBuffer<MLFloat16> BufferAFp16;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<uint8_t> BufferQuantBData;
  MatrixGuardBuffer<float> BufferQuantBScale;
  MatrixGuardBuffer<uint8_t> BufferQuantBZeroPoint;
  MatrixGuardBuffer<std::byte> BufferPackedQuantBData;
  MatrixGuardBuffer<std::byte> BufferWorkspaceRef;
  MatrixGuardBuffer<std::byte> BufferWorkspaceTest;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferCRef;
  MatrixGuardBuffer<float> BufferCTest;
  MatrixGuardBuffer<MLFloat16> BufferCRefFp16;
  MatrixGuardBuffer<MLFloat16> BufferCTestFp16;

  void RunOne(size_t M, size_t N, size_t K, bool Symmetric, bool WithBias, bool WithThreadpool) {
    MLAS_THREADPOOL* Threadpool = WithThreadpool ? GetMlasThreadPool() : nullptr;

    // fp16 activations plus the same values widened to float. The float buffer is
    // exactly what the operator's bulk fp16 -> fp32 conversion of A produces, so the
    // two GEMM runs below see numerically identical A.
    MLFloat16* AFp16 = BufferAFp16.GetBuffer(M * K);
    float* AFloat = BufferAFloat.GetBuffer(M * K);
    std::mt19937 gen(static_cast<unsigned>(M * 31 + N * 17 + K * 7 + BlkLen));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < M * K; ++i) {
      MLFloat16 h(dist(gen));
      AFp16[i] = h;
      AFloat[i] = h.ToFloat();
    }

    float* B = BufferB.GetBuffer(N * K);
    for (size_t i = 0; i < N * K; ++i) {
      B[i] = dist(gen);
    }

    const float* Bias = nullptr;
    if (WithBias) {
      float* bias = BufferBias.GetBuffer(N);
      for (size_t i = 0; i < N; ++i) {
        bias[i] = dist(gen);
      }
      Bias = bias;
    }

    size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
    MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(static_cast<int>(BlkLen), /* columnwise */ true,
                                                   static_cast<int>(K), static_cast<int>(N),
                                                   QuantBDataSizeInBytes, QuantBScaleSize,
                                                   &QuantBZeroPointSizeInBytes);

    uint8_t* QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
    float* QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);
    uint8_t* QuantBZeroPoint = Symmetric ? nullptr : BufferQuantBZeroPoint.GetBuffer(QuantBZeroPointSizeInBytes);

    MlasQuantizeBlockwise<float, BlkBitWidth>(QuantBData, QuantBScale, QuantBZeroPoint,
                                              B, static_cast<int>(BlkLen), /* columnwise */ true,
                                              static_cast<int>(K), static_cast<int>(N),
                                              static_cast<int>(N), GetMlasThreadPool());

    void* PackedQuantBDataWorkspace = nullptr;
    if (const auto PackedQuantBDataSize =
            MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, !Symmetric, SQNBIT_CompInt8, nullptr);
        PackedQuantBDataSize > 0) {
      PackedQuantBDataWorkspace = BufferPackedQuantBData.GetBuffer(PackedQuantBDataSize);
      MlasQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, SQNBIT_CompInt8, QuantBData, PackedQuantBDataWorkspace,
                                  QuantBScale, QuantBZeroPoint != nullptr, QuantBZeroPoint,
                                  GetMlasThreadPool(), nullptr);
    }

    void* WorkspaceRef = nullptr;
    void* WorkspaceTest = nullptr;
    if (const auto WorkspaceSize =
            MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen, !Symmetric, SQNBIT_CompInt8, nullptr);
        WorkspaceSize > 0) {
      WorkspaceRef = BufferWorkspaceRef.GetBuffer(WorkspaceSize);
      WorkspaceTest = BufferWorkspaceTest.GetBuffer(WorkspaceSize);
    }

    float* CRef = BufferCRef.GetBuffer(M * N, true);
    float* CTest = BufferCTest.GetBuffer(M * N, true);

    auto run = [&](const float* Afloat, const MLFloat16* Afp16, float* C, MLFloat16* Cfp16, void* Workspace) {
      MLAS_QNBIT_GEMM_DATA_PARAMS<float> params;
      params.A = Afloat;
      params.AFp16 = Afp16;
      params.lda = K;
      params.Bias = Bias;
      params.C = C;
      params.CFp16 = Cfp16;
      params.ldc = N;
#ifdef MLAS_TARGET_AMD64_IX86
      params.QuantBDataWorkspace = PackedQuantBDataWorkspace;
#endif
      params.PackedQuantBData = static_cast<const std::byte*>(PackedQuantBDataWorkspace);
      params.QuantBScale = QuantBScale;
      params.QuantBZeroPoint = QuantBZeroPoint;
      MlasQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, SQNBIT_CompInt8, &params, Workspace, Threadpool, nullptr);
    };

    run(AFloat, nullptr, CRef, nullptr, WorkspaceRef);   // reference: A converted to fp32 first
    run(nullptr, AFp16, CTest, nullptr, WorkspaceTest);  // native A quantize, fp32 output

    for (size_t i = 0; i < M * N; ++i) {
      ASSERT_EQ(CTest[i], CRef[i])
          << "A-path mismatch at " << i << " (M=" << M << ", N=" << N << ", K=" << K << ", BlkLen=" << BlkLen
          << ", symmetric=" << Symmetric << ", bias=" << WithBias << ", threadpool=" << WithThreadpool << ")";
    }

    // fp16 output path: the direct fp16 C must equal the fp32 reference converted to fp16.
    if (MlasQNBitGemmFp16DirectCOutputSupported(BlkBitWidth)) {
      MLFloat16* CRefFp16 = BufferCRefFp16.GetBuffer(M * N, true);
      MLFloat16* CTestFp16 = BufferCTestFp16.GetBuffer(M * N, true);
      MlasConvertFloatToHalfBuffer(CRef, CRefFp16, M * N);
      run(nullptr, AFp16, nullptr, CTestFp16, WorkspaceTest);  // native A quantize, fp16 output

      for (size_t i = 0; i < M * N; ++i) {
        ASSERT_EQ(CTestFp16[i].val, CRefFp16[i].val)
            << "C-path mismatch at " << i << " (M=" << M << ", N=" << N << ", K=" << K << ", BlkLen=" << BlkLen
            << ", symmetric=" << Symmetric << ", bias=" << WithBias << ", threadpool=" << WithThreadpool << ")";
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name = std::string("QNBitGemmFp16QuantA") + "BlkLen" + std::to_string(BlkLen);
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    if (!MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, SQNBIT_CompInt8) ||
        !MlasQNBitGemmFp16DirectQuantASupported()) {
      return;
    }
    for (bool threadpool : {false, true}) {
      for (bool symmetric : {true, false}) {
        for (bool bias : {false, true}) {
          RunOne(1, 32, 64, symmetric, bias, threadpool);
          RunOne(4, 33, 65, symmetric, bias, threadpool);
          RunOne(16, 64, 128, symmetric, bias, threadpool);
          RunOne(43, 500, 401, symmetric, bias, threadpool);
          RunOne(129, 257, 1031, symmetric, bias, threadpool);
        }
      }
    }
  }
};

static size_t QNBitGemmFp16QuantARegisterAll() {
  size_t count = 0;
  count += MlasDirectShortExecuteTests<MlasQNBitGemmFp16QuantATest<16>>::RegisterShortExecute();
  count += MlasDirectShortExecuteTests<MlasQNBitGemmFp16QuantATest<32>>::RegisterShortExecute();
  count += MlasDirectShortExecuteTests<MlasQNBitGemmFp16QuantATest<64>>::RegisterShortExecute();
  count += MlasDirectShortExecuteTests<MlasQNBitGemmFp16QuantATest<128>>::RegisterShortExecute();
  count += MlasDirectShortExecuteTests<MlasQNBitGemmFp16QuantATest<256>>::RegisterShortExecute();
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return QNBitGemmFp16QuantARegisterAll();
      }
      return 0;
    });

#endif  // ORT_MINIMAL_BUILD
