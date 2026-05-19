// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "mlas_qkv_quant.h"

#include <cmath>
#include <random>
#include <string>

//
// Tests for MlasKVQuantize / MlasKVDequantize roundtrip and
// MlasQKGemm / MlasSVGemm correctness against FP32 SGEMM oracle.
//

class MlasKVQuantTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferA;
  MatrixGuardBuffer<float> BufferB;
  MatrixGuardBuffer<float> BufferC;
  MatrixGuardBuffer<float> BufferCRef;
  MatrixGuardBuffer<float> BufferBDequant;
  MatrixGuardBuffer<float> BufferScales;
  MatrixGuardBuffer<uint8_t> BufferQuantized;

  void FillRandom(float* buf, size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; i++) {
      buf[i] = dist(gen);
    }
  }

  bool IsInt4(MLAS_KV_QUANT_TYPE qt) {
    return qt == MLAS_KV_QUANT_TYPE::S4_PerTensor || qt == MLAS_KV_QUANT_TYPE::S4_PerChannel;
  }

  bool IsPerChannel(MLAS_KV_QUANT_TYPE qt) {
    return qt == MLAS_KV_QUANT_TYPE::S8_PerChannel || qt == MLAS_KV_QUANT_TYPE::S4_PerChannel;
  }

  void ComputeScales(const float* data, size_t rows, size_t cols, MLAS_KV_QUANT_TYPE qt, float* scales) {
    float qmax = IsInt4(qt) ? 7.0f : 127.0f;
    if (IsPerChannel(qt)) {
      for (size_t c = 0; c < cols; c++) {
        float amax = 0.0f;
        for (size_t r = 0; r < rows; r++) {
          amax = std::max(amax, std::fabs(data[r * cols + c]));
        }
        scales[c] = (amax > 1e-6f) ? (amax / qmax) : 1.0f;
      }
    } else {
      float amax = 0.0f;
      for (size_t i = 0; i < rows * cols; i++) {
        amax = std::max(amax, std::fabs(data[i]));
      }
      scales[0] = (amax > 1e-6f) ? (amax / qmax) : 1.0f;
    }
  }

  //
  // Test: quantize -> dequantize roundtrip within tolerance.
  //
  void TestRoundtrip(size_t Rows, size_t Cols, MLAS_KV_QUANT_TYPE QuantType) {
    const size_t num_scales = IsPerChannel(QuantType) ? Cols : 1;
    float* src = BufferA.GetBuffer(Rows * Cols);
    float* scales = BufferScales.GetBuffer(num_scales);
    float* dst = BufferBDequant.GetBuffer(Rows * Cols);

    FillRandom(src, Rows * Cols, static_cast<unsigned>(Rows * 1000 + Cols));
    ComputeScales(src, Rows, Cols, QuantType, scales);

    size_t packed_bytes = Rows * MlasKVQuantPackedRowBytes(QuantType, Cols);
    uint8_t* quant_buf = BufferQuantized.GetBuffer(packed_bytes);

    MlasKVQuantize(src, quant_buf, Rows, Cols, Cols, QuantType, scales, nullptr);
    MlasKVDequantize(quant_buf, dst, Rows, Cols, Cols, QuantType, scales, nullptr);

    // Tolerance: INT8 loses ~1/127 of range, INT4 loses ~1/7
    float atol = IsInt4(QuantType) ? 0.3f : 0.02f;
    for (size_t i = 0; i < Rows * Cols; i++) {
      float diff = std::fabs(src[i] - dst[i]);
      ASSERT_LE(diff, atol)
          << "Roundtrip mismatch at [" << i / Cols << ", " << i % Cols
          << "], src=" << src[i] << " dst=" << dst[i]
          << " qt=" << static_cast<int>(QuantType);
    }
  }

  //
  // Reference FP32 SGEMM: C = alpha * A * B^T
  //
  void RefQKGemm(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K, float alpha, size_t lda, size_t ldc) {
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++) {
          acc += A[m * lda + k] * B[n * K + k];  // B is [N, K] row-major, transpose
        }
        C[m * ldc + n] = alpha * acc;
      }
    }
  }

  //
  // Reference FP32 SGEMM: C = A * B (no transpose)
  //
  void RefSVGemm(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K, size_t lda, size_t ldc) {
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++) {
          acc += A[m * lda + k] * B[k * N + n];  // B is [K, N] row-major
        }
        C[m * ldc + n] = acc;
      }
    }
  }

  //
  // Test: MlasQKGemm correctness vs FP32 SGEMM oracle.
  //
  void TestQKGemm(size_t M, size_t N, size_t K, MLAS_KV_QUANT_TYPE QuantType) {
    const size_t num_scales = IsPerChannel(QuantType) ? K : 1;
    float* A = BufferA.GetBuffer(M * K);
    float* B = BufferB.GetBuffer(N * K);
    float* scales = BufferScales.GetBuffer(num_scales);
    float* C = BufferC.GetBuffer(M * N);
    float* CRef = BufferCRef.GetBuffer(M * N);
    float* BDequant = BufferBDequant.GetBuffer(N * K);

    unsigned seed = static_cast<unsigned>(M * 10000 + N * 100 + K);
    FillRandom(A, M * K, seed);
    FillRandom(B, N * K, seed + 1);
    ComputeScales(B, N, K, QuantType, scales);

    // Quantize B
    size_t packed_bytes = N * MlasKVQuantPackedRowBytes(QuantType, K);
    uint8_t* BQuant = BufferQuantized.GetBuffer(packed_bytes);
    MlasKVQuantize(B, BQuant, N, K, K, QuantType, scales, nullptr);

    // Dequantize B for reference GEMM
    MlasKVDequantize(BQuant, BDequant, N, K, K, QuantType, scales, nullptr);

    float alpha = 1.0f / std::sqrt(static_cast<float>(K));

    // Reference: alpha * A * BDequant^T
    RefQKGemm(A, BDequant, CRef, M, N, K, alpha, K, N);

    // Quantized: MlasQKGemm
    MlasQKGemm(M, N, K, alpha, A, K, BQuant, QuantType, scales, C, N, nullptr);

    float atol = IsInt4(QuantType) ? 0.15f : 0.02f;
    float rtol = IsInt4(QuantType) ? 0.1f : 0.01f;
    for (size_t i = 0; i < M * N; i++) {
      float diff = std::fabs(C[i] - CRef[i]);
      float threshold = atol + rtol * std::fabs(CRef[i]);
      ASSERT_LE(diff, threshold)
          << "QKGemm mismatch at [" << i / N << ", " << i % N
          << "], got=" << C[i] << " ref=" << CRef[i]
          << " M=" << M << " N=" << N << " K=" << K
          << " qt=" << static_cast<int>(QuantType);
    }
  }

  //
  // Test: MlasSVGemm correctness vs FP32 SGEMM oracle.
  //
  void TestSVGemm(size_t M, size_t N, size_t K, MLAS_KV_QUANT_TYPE QuantType) {
    const size_t num_scales = IsPerChannel(QuantType) ? N : 1;
    float* A = BufferA.GetBuffer(M * K);
    float* B = BufferB.GetBuffer(K * N);
    float* scales = BufferScales.GetBuffer(num_scales);
    float* C = BufferC.GetBuffer(M * N);
    float* CRef = BufferCRef.GetBuffer(M * N);
    float* BDequant = BufferBDequant.GetBuffer(K * N);

    unsigned seed = static_cast<unsigned>(M * 10000 + N * 100 + K + 7);
    FillRandom(A, M * K, seed);
    // For SV, A is attention probabilities (non-negative, sum to ~1 per row)
    for (size_t m = 0; m < M; m++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        A[m * K + k] = std::fabs(A[m * K + k]);
        sum += A[m * K + k];
      }
      if (sum > 0.0f) {
        for (size_t k = 0; k < K; k++) {
          A[m * K + k] /= sum;
        }
      }
    }
    FillRandom(B, K * N, seed + 1);
    ComputeScales(B, K, N, QuantType, scales);

    // Quantize B
    size_t packed_bytes = K * MlasKVQuantPackedRowBytes(QuantType, N);
    uint8_t* BQuant = BufferQuantized.GetBuffer(packed_bytes);
    MlasKVQuantize(B, BQuant, K, N, N, QuantType, scales, nullptr);

    // Dequantize B for reference GEMM
    MlasKVDequantize(BQuant, BDequant, K, N, N, QuantType, scales, nullptr);

    // Reference: A * BDequant
    RefSVGemm(A, BDequant, CRef, M, N, K, K, N);

    // Quantized: MlasSVGemm
    MlasSVGemm(M, N, K, A, K, BQuant, QuantType, scales, C, N, nullptr);

    float atol = IsInt4(QuantType) ? 0.15f : 0.02f;
    float rtol = IsInt4(QuantType) ? 0.1f : 0.01f;
    for (size_t i = 0; i < M * N; i++) {
      float diff = std::fabs(C[i] - CRef[i]);
      float threshold = atol + rtol * std::fabs(CRef[i]);
      ASSERT_LE(diff, threshold)
          << "SVGemm mismatch at [" << i / N << ", " << i % N
          << "], got=" << C[i] << " ref=" << CRef[i]
          << " M=" << M << " N=" << N << " K=" << K
          << " qt=" << static_cast<int>(QuantType);
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("KVQuant");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    static const MLAS_KV_QUANT_TYPE AllQuantTypes[] = {
        MLAS_KV_QUANT_TYPE::S8_PerTensor,
        MLAS_KV_QUANT_TYPE::S8_PerChannel,
        MLAS_KV_QUANT_TYPE::S4_PerTensor,
        MLAS_KV_QUANT_TYPE::S4_PerChannel,
    };

    // Roundtrip tests
    for (auto qt : AllQuantTypes) {
      // INT4 requires even column count
      for (size_t rows : {size_t{1}, size_t{4}, size_t{16}, size_t{64}}) {
        for (size_t cols : {IsInt4(qt) ? size_t{2} : size_t{1}, size_t{8}, size_t{32}, size_t{64}, size_t{128}}) {
          TestRoundtrip(rows, cols, qt);
        }
      }
    }

    // QKGemm tests: C[M,N] = alpha * A[M,K] * B^T[K,N]
    for (auto qt : AllQuantTypes) {
      for (size_t M : {size_t{1}, size_t{4}, size_t{16}}) {                                // seq_length
        for (size_t N : {size_t{1}, size_t{8}, size_t{32}, size_t{128}}) {                 // total_seqlen
          for (size_t K : {IsInt4(qt) ? size_t{2} : size_t{1}, size_t{32}, size_t{64}}) {  // head_size
            TestQKGemm(M, N, K, qt);
          }
        }
      }
    }

    // SVGemm tests: C[M,N] = A[M,K] * B[K,N]
    for (auto qt : AllQuantTypes) {
      for (size_t M : {size_t{1}, size_t{4}, size_t{16}}) {                                // seq_length
        for (size_t K : {size_t{1}, size_t{8}, size_t{32}, size_t{128}}) {                 // total_seqlen
          for (size_t N : {IsInt4(qt) ? size_t{2} : size_t{1}, size_t{32}, size_t{64}}) {  // head_size
            TestSVGemm(M, N, K, qt);
          }
        }
      }
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasKVQuantTest>::RegisterShortExecute();
  }
  return count;
});
