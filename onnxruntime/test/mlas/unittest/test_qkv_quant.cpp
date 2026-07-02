// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "mlas_qkv_quant.h"
#include "core/platform/env_var.h"

#include <algorithm>
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

  void TestQKGemmS8PerTensorPreservesFp32Query() {
    if (onnxruntime::detail::GetEnvironmentVar("ORT_MLAS_QKGEMM_S8_APPROX_VNNI") == "1") {
      return;
    }

    constexpr size_t M = 1;
    constexpr size_t N = 3;
    constexpr size_t K = 128;
    constexpr MLAS_KV_QUANT_TYPE QuantType = MLAS_KV_QUANT_TYPE::S8_PerTensor;

    float* A = BufferA.GetBuffer(M * K);
    float* B = BufferB.GetBuffer(N * K);
    float* scales = BufferScales.GetBuffer(1);
    float* C = BufferC.GetBuffer(M * N);
    float* CRef = BufferCRef.GetBuffer(M * N);
    float* BDequant = BufferBDequant.GetBuffer(N * K);

    A[0] = 1024.0f;
    for (size_t k = 1; k < K; ++k) {
      A[k] = 3.0f;
    }

    for (size_t i = 0; i < N * K; ++i) {
      B[i] = 1.0f;
    }
    ComputeScales(B, N, K, QuantType, scales);

    size_t packed_bytes = N * MlasKVQuantPackedRowBytes(QuantType, K);
    uint8_t* BQuant = BufferQuantized.GetBuffer(packed_bytes);
    MlasKVQuantize(B, BQuant, N, K, K, QuantType, scales, nullptr);
    MlasKVDequantize(BQuant, BDequant, N, K, K, QuantType, scales, nullptr);

    float alpha = 1.0f / std::sqrt(static_cast<float>(K));
    RefQKGemm(A, BDequant, CRef, M, N, K, alpha, K, N);
    MlasQKGemm(M, N, K, alpha, A, K, BQuant, QuantType, scales, C, N, nullptr);

    for (size_t i = 0; i < M * N; i++) {
      ASSERT_NEAR(C[i], CRef[i], 1e-3f)
          << "QKGemm S8 per-tensor must preserve FP32 query values at output " << i;
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
    MlasSVGemm(M, N, K, A, K, BQuant, QuantType, scales, C, N, 0.0f, nullptr);

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
    TestQKGemmS8PerTensorPreservesFp32Query();
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

//
// Focused test for MlasFlashAttentionQuantizedKV:
// Validates the tiled online-softmax kernel against a naive reference pipeline
// (MlasQKGemm + softmax + MlasSVGemm) across INT8/INT4, per-tensor/per-channel.
//
class MlasFlashAttentionQuantizedKVTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferQ;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputRef;
  MatrixGuardBuffer<float> BufferScores;
  MatrixGuardBuffer<float> BufferProbs;
  MatrixGuardBuffer<float> BufferScalesK;
  MatrixGuardBuffer<float> BufferScalesV;
  MatrixGuardBuffer<float> BufferKFP32;
  MatrixGuardBuffer<float> BufferVFP32;
  MatrixGuardBuffer<float> BufferFlash;
  MatrixGuardBuffer<float> BufferPartials;
  MatrixGuardBuffer<uint8_t> BufferKQuant;
  MatrixGuardBuffer<uint8_t> BufferVQuant;

  void FillRandom(float* buf, size_t n, unsigned seed, float lo = -0.5f, float hi = 0.5f) {
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

  // Naive reference: for a single (batch=1, head=1) attention computation
  // Q[seq_len, head_size], K[total_seqlen, head_size], V[total_seqlen, head_size]
  // -> output[seq_len, head_size]
  // Uses quantized K/V via MlasQKGemm + softmax + MlasSVGemm.
  void NaiveReference(
      const float* Q, size_t seq_len, size_t total_seqlen, size_t head_size,
      const uint8_t* k_quant, const uint8_t* v_quant,
      MLAS_KV_QUANT_TYPE quant_type, const float* k_scale, const float* v_scale,
      float scale, int past_seqlen, float* output) {
    float* scores = BufferScores.GetBuffer(seq_len * total_seqlen);
    float* probs = BufferProbs.GetBuffer(seq_len * total_seqlen);

    // QK^T
    MlasQKGemm(seq_len, total_seqlen, head_size, scale,
               Q, head_size, k_quant, quant_type, k_scale,
               scores, total_seqlen, nullptr);

    // Causal mask + softmax
    for (size_t q_s = 0; q_s < seq_len; q_s++) {
      size_t causal_limit = static_cast<size_t>(past_seqlen) + q_s + 1;
      // Apply causal mask
      for (size_t kv_s = 0; kv_s < total_seqlen; kv_s++) {
        if (kv_s >= causal_limit) {
          scores[q_s * total_seqlen + kv_s] = -std::numeric_limits<float>::infinity();
        }
      }
      // Softmax
      float max_val = -std::numeric_limits<float>::infinity();
      for (size_t kv_s = 0; kv_s < total_seqlen; kv_s++) {
        max_val = std::max(max_val, scores[q_s * total_seqlen + kv_s]);
      }
      float sum_exp = 0.0f;
      for (size_t kv_s = 0; kv_s < total_seqlen; kv_s++) {
        probs[q_s * total_seqlen + kv_s] = std::exp(scores[q_s * total_seqlen + kv_s] - max_val);
        sum_exp += probs[q_s * total_seqlen + kv_s];
      }
      for (size_t kv_s = 0; kv_s < total_seqlen; kv_s++) {
        probs[q_s * total_seqlen + kv_s] /= sum_exp;
      }
    }

    // SV GEMM
    MlasSVGemm(seq_len, head_size, total_seqlen,
               probs, total_seqlen, v_quant, quant_type, v_scale,
               output, head_size, 0.0f, nullptr);
  }

  void TestFlashAttention(size_t seq_len, size_t total_seqlen, size_t head_size,
                          MLAS_KV_QUANT_TYPE quant_type) {
    const size_t k_num_scales = IsPerChannel(quant_type) ? head_size : 1;
    const size_t v_num_scales = IsPerChannel(quant_type) ? head_size : 1;
    const size_t packed_row_bytes = MlasKVQuantPackedRowBytes(quant_type, head_size);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const int past_seqlen = static_cast<int>(total_seqlen - seq_len);

    // Allocate and fill
    float* Q = BufferQ.GetBuffer(seq_len * head_size);
    float* K_fp32 = BufferKFP32.GetBuffer(total_seqlen * head_size);
    float* V_fp32 = BufferVFP32.GetBuffer(total_seqlen * head_size);
    float* k_scale = BufferScalesK.GetBuffer(k_num_scales);
    float* v_scale = BufferScalesV.GetBuffer(v_num_scales);
    float* output_flash = BufferOutput.GetBuffer(seq_len * head_size);
    float* output_ref = BufferOutputRef.GetBuffer(seq_len * head_size);

    unsigned seed = static_cast<unsigned>(seq_len * 1000 + total_seqlen * 10 + head_size);
    FillRandom(Q, seq_len * head_size, seed);
    FillRandom(K_fp32, total_seqlen * head_size, seed + 1);
    FillRandom(V_fp32, total_seqlen * head_size, seed + 2);

    ComputeScales(K_fp32, total_seqlen, head_size, quant_type, k_scale);
    ComputeScales(V_fp32, total_seqlen, head_size, quant_type, v_scale);

    // Quantize K and V
    uint8_t* k_quant = BufferKQuant.GetBuffer(total_seqlen * packed_row_bytes);
    uint8_t* v_quant = BufferVQuant.GetBuffer(total_seqlen * packed_row_bytes);
    MlasKVQuantize(K_fp32, k_quant, total_seqlen, head_size, head_size, quant_type, k_scale, nullptr);
    MlasKVQuantize(V_fp32, v_quant, total_seqlen, head_size, head_size, quant_type, v_scale, nullptr);

    // Naive reference
    NaiveReference(Q, seq_len, total_seqlen, head_size,
                   k_quant, v_quant, quant_type, k_scale, v_scale,
                   scale, past_seqlen, output_ref);

    // Flash attention
    int q_block_size = std::min(static_cast<int>(seq_len), 16);
    int kv_block_size = std::min(static_cast<int>(total_seqlen), 32);

    size_t buffer_size_per_thread =
        (static_cast<size_t>(q_block_size) * 2 +
         static_cast<size_t>(q_block_size) * static_cast<size_t>(kv_block_size) +
         static_cast<size_t>(q_block_size) * static_cast<size_t>(head_size)) *
        sizeof(float);
    float* flash_buffer = BufferFlash.GetBuffer(buffer_size_per_thread / sizeof(float));

    MlasFlashAttentionQuantizedKVArgs args;
    args.batch_size = 1;
    args.num_heads = 1;
    args.kv_num_heads = 1;
    args.sequence_length = static_cast<int>(seq_len);
    args.total_seqlen = static_cast<int>(total_seqlen);
    args.head_size = static_cast<int>(head_size);
    args.past_seqlen = past_seqlen;
    args.local_window_size = -1;
    args.seqlen_present_kv = static_cast<int>(total_seqlen);
    args.q_block_size = q_block_size;
    args.kv_block_size = kv_block_size;
    args.scale = scale;
    args.quant_type = quant_type;
    args.per_channel_k = IsPerChannel(quant_type);
    args.per_channel_v = IsPerChannel(quant_type);
    args.thread_count = 1;
    args.buffer = flash_buffer;
    args.buffer_size_per_thread = buffer_size_per_thread;
    args.query = Q;
    args.q_batch_stride = seq_len * head_size;
    args.k_cache = k_quant;
    args.v_cache = v_quant;
    args.k_scale = k_scale;
    args.v_scale = v_scale;
    args.output = output_flash;
    args.attention_bias = nullptr;
    args.attention_bias_seqlen_stride = 0;
    args.attention_bias_broadcast_batch = true;
    args.attention_bias_broadcast_head = true;
    args.flash_decoding_partials = nullptr;
    args.kv_chunk_count = 0;

    MlasFlashAttentionQuantizedKV(&args, nullptr);

    // Compare: flash uses ComputeSumExpF32Kernel (SIMD polynomial approx) while
    // NaiveReference uses std::exp. Tolerance accounts for accumulation order
    // differences across platforms/ISAs.
    float atol = IsInt4(quant_type) ? 1e-3f : 1e-4f;
    for (size_t i = 0; i < seq_len * head_size; i++) {
      float diff = std::fabs(output_flash[i] - output_ref[i]);
      ASSERT_LE(diff, atol)
          << "FlashAttention vs Naive mismatch at [" << i / head_size << ", " << i % head_size
          << "], flash=" << output_flash[i] << " ref=" << output_ref[i]
          << " seq_len=" << seq_len << " total_seqlen=" << total_seqlen
          << " head_size=" << head_size
          << " qt=" << static_cast<int>(quant_type);
    }
  }

  // Test flash decoding path: sequence_length=1 with KV split across chunks
  void TestFlashDecoding(size_t total_seqlen, size_t head_size,
                         MLAS_KV_QUANT_TYPE quant_type) {
    const size_t seq_len = 1;
    const size_t k_num_scales = IsPerChannel(quant_type) ? head_size : 1;
    const size_t v_num_scales = IsPerChannel(quant_type) ? head_size : 1;
    const size_t packed_row_bytes = MlasKVQuantPackedRowBytes(quant_type, head_size);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const int past_seqlen = static_cast<int>(total_seqlen - 1);

    // Allocate and fill
    float* Q = BufferQ.GetBuffer(head_size);
    float* K_fp32 = BufferKFP32.GetBuffer(total_seqlen * head_size);
    float* V_fp32 = BufferVFP32.GetBuffer(total_seqlen * head_size);
    float* k_scale_buf = BufferScalesK.GetBuffer(k_num_scales);
    float* v_scale_buf = BufferScalesV.GetBuffer(v_num_scales);
    float* output_flash = BufferOutput.GetBuffer(head_size);
    float* output_ref = BufferOutputRef.GetBuffer(head_size);

    unsigned seed = static_cast<unsigned>(total_seqlen * 100 + head_size * 7);
    FillRandom(Q, head_size, seed);
    FillRandom(K_fp32, total_seqlen * head_size, seed + 1);
    FillRandom(V_fp32, total_seqlen * head_size, seed + 2);

    ComputeScales(K_fp32, total_seqlen, head_size, quant_type, k_scale_buf);
    ComputeScales(V_fp32, total_seqlen, head_size, quant_type, v_scale_buf);

    // Quantize K and V
    uint8_t* k_quant = BufferKQuant.GetBuffer(total_seqlen * packed_row_bytes);
    uint8_t* v_quant = BufferVQuant.GetBuffer(total_seqlen * packed_row_bytes);
    MlasKVQuantize(K_fp32, k_quant, total_seqlen, head_size, head_size, quant_type, k_scale_buf, nullptr);
    MlasKVQuantize(V_fp32, v_quant, total_seqlen, head_size, head_size, quant_type, v_scale_buf, nullptr);

    // Naive reference
    NaiveReference(Q, seq_len, total_seqlen, head_size,
                   k_quant, v_quant, quant_type, k_scale_buf, v_scale_buf,
                   scale, past_seqlen, output_ref);

    // Flash decoding: use small kv_block_size to get multiple chunks
    int kv_block_size = std::min(static_cast<int>(total_seqlen), 16);
    int kv_chunk_count = (static_cast<int>(total_seqlen) + kv_block_size - 1) / kv_block_size;

    // Per-thread scratch: scores[kv_block_size]
    size_t buffer_size_per_thread = static_cast<size_t>(kv_block_size) * sizeof(float);
    float* flash_buffer = BufferFlash.GetBuffer(buffer_size_per_thread / sizeof(float));

    // Partials buffer: [1 batch * 1 head * kv_chunk_count * (2 + head_size)]
    size_t partials_count = static_cast<size_t>(kv_chunk_count) * (2 + head_size);
    float* partials = BufferPartials.GetBuffer(partials_count);

    MlasFlashAttentionQuantizedKVArgs args;
    args.batch_size = 1;
    args.num_heads = 1;
    args.kv_num_heads = 1;
    args.sequence_length = 1;
    args.total_seqlen = static_cast<int>(total_seqlen);
    args.head_size = static_cast<int>(head_size);
    args.past_seqlen = past_seqlen;
    args.local_window_size = -1;
    args.seqlen_present_kv = static_cast<int>(total_seqlen);
    args.q_block_size = 1;
    args.kv_block_size = kv_block_size;
    args.scale = scale;
    args.quant_type = quant_type;
    args.per_channel_k = IsPerChannel(quant_type);
    args.per_channel_v = IsPerChannel(quant_type);
    args.thread_count = 1;
    args.buffer = flash_buffer;
    args.buffer_size_per_thread = buffer_size_per_thread;
    args.query = Q;
    args.q_batch_stride = head_size;
    args.k_cache = k_quant;
    args.v_cache = v_quant;
    args.k_scale = k_scale_buf;
    args.v_scale = v_scale_buf;
    args.output = output_flash;
    args.attention_bias = nullptr;
    args.attention_bias_seqlen_stride = 0;
    args.attention_bias_broadcast_batch = true;
    args.attention_bias_broadcast_head = true;
    args.flash_decoding_partials = partials;
    args.kv_chunk_count = kv_chunk_count;

    MlasFlashAttentionQuantizedKV(&args, nullptr);

    // Compare: flash decoding has an extra reduce phase with exp rescaling,
    // so tolerance is slightly larger than the single-pass flash attention test.
    float atol = IsInt4(quant_type) ? 1e-3f : 1e-4f;
    for (size_t i = 0; i < head_size; i++) {
      float diff = std::fabs(output_flash[i] - output_ref[i]);
      ASSERT_LE(diff, atol)
          << "FlashDecoding vs Naive mismatch at [" << i
          << "], flash=" << output_flash[i] << " ref=" << output_ref[i]
          << " total_seqlen=" << total_seqlen
          << " head_size=" << head_size
          << " qt=" << static_cast<int>(quant_type);
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("FlashAttentionQuantizedKV");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    static const MLAS_KV_QUANT_TYPE AllQuantTypes[] = {
        MLAS_KV_QUANT_TYPE::S8_PerTensor,
        MLAS_KV_QUANT_TYPE::S8_PerChannel,
        MLAS_KV_QUANT_TYPE::S4_PerTensor,
        MLAS_KV_QUANT_TYPE::S4_PerChannel,
    };

    for (auto qt : AllQuantTypes) {
      size_t min_head = size_t{4};
      for (size_t seq_len : {size_t{1}, size_t{4}, size_t{16}}) {
        for (size_t total_seqlen : {size_t{4}, size_t{32}, size_t{64}}) {
          if (total_seqlen < seq_len) continue;
          for (size_t head_size : {min_head, size_t{32}, size_t{64}}) {
            TestFlashAttention(seq_len, total_seqlen, head_size, qt);
          }
        }
      }
      // Flash decoding tests (sequence_length=1 with KV split into chunks)
      for (size_t total_seqlen : {size_t{4}, size_t{32}, size_t{64}, size_t{128}}) {
        for (size_t head_size : {min_head, size_t{32}, size_t{64}}) {
          TestFlashDecoding(total_seqlen, head_size, qt);
        }
      }
    }
  }
};

static UNUSED_VARIABLE bool added_flash_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasFlashAttentionQuantizedKVTest>::RegisterShortExecute();
  }
  return count;
});
