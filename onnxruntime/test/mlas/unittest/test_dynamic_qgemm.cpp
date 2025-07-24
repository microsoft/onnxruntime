//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "test_util.h"

class MlasDynamicQgemmTest {
 public:
  void Test(size_t M, size_t N, size_t K, size_t BatchSize) {
    // Setup buffers for holding various data
    MatrixGuardBuffer<float> buffer_a;
    MatrixGuardBuffer<float> buffer_bf;
    MatrixGuardBuffer<int8_t> buffer_bq;
    MatrixGuardBuffer<float> buffer_c;
    MatrixGuardBuffer<float> buffer_c_ref;

    float* A = buffer_a.GetBuffer(M * K * BatchSize);
    // Buffer for holding floating point version of weight matrix
    float* Bf = buffer_bf.GetBuffer(K * N * BatchSize);
    // Buffer for holding quantized version of weight matrix
    int8_t* Bq = buffer_bq.GetBuffer(K * N * BatchSize);
    float* C = buffer_c.GetBuffer(M * N * BatchSize);
    float* CRef = buffer_c_ref.GetBuffer(M * N * BatchSize);

    // Initialize A and Bf
    for (size_t i = 0; i < M * K * BatchSize; ++i)
      A[i] = static_cast<float>((rand() % 255 - 128) / 16.0f);
    for (size_t i = 0; i < K * N * BatchSize; ++i)
      Bf[i] = static_cast<float>((rand() % 255 - 128) / 16.0f);

    // Quantize Bf → Bq and compute per-column scale and bias per batch
    std::vector<std::vector<float>> b_scale_batches(BatchSize, std::vector<float>(N));
    std::vector<std::vector<float>> b_bias_batches(BatchSize, std::vector<float>(N, 0.0f));

    for (size_t b = 0; b < BatchSize; ++b) {
      for (size_t n = 0; n < N; ++n) {
        float min_val = Bf[b * K * N + n];
        float max_val = min_val;
        for (size_t k = 1; k < K; ++k) {
          float v = Bf[b * K * N + k * N + n];
          min_val = std::min(min_val, v);
          max_val = std::max(max_val, v);
        }
        float scale = (max_val - min_val) / 255.0f;
        if (scale < 1e-8f) scale = 1.0f;
        b_scale_batches[b][n] = scale;

        for (size_t k = 0; k < K; ++k) {
          float v = Bf[b * K * N + k * N + n];
          int q = static_cast<int>(std::round(v / scale));
          Bq[b * K * N + k * N + n] = static_cast<int8_t>(std::clamp(q, -128, 127));
        }
      }
    }

    // Prepare kernel parameters
    MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS shape{M, N, K};
    std::vector<uint8_t> packed_b_storage(BatchSize * MlasDynamicQgemmPackBSize(N, K));
    std::vector<MLAS_GEMM_DYN_QUANT_DATA_PARAMS> params(BatchSize);

    for (size_t b = 0; b < BatchSize; ++b) {
      params[b].A = A + b * M * K;
      params[b].lda = K;
      params[b].ldb = N;
      params[b].C = C + b * M * N;
      params[b].ldc = N;
      // Pack b matrix using MlasDynamicQgemmPackBSize & MlasDynamicQgemmPackB
      void* packed_b = packed_b_storage.data() + b * MlasDynamicQgemmPackBSize(N, K);
      MlasDynamicQgemmPackB(N, K,
                            Bq + b * K * N,
                            b_scale_batches[b].data(),
                            b_bias_batches[b].data(),
                            packed_b);
      params[b].PackedB = packed_b;
    }

    // call MlasDynamicQGemmBatch Function
    MlasDynamicQGemmBatch(shape, params.data(), BatchSize, nullptr);

    // Compute reference result
    for (size_t b = 0; b < BatchSize; ++b) {
      for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
          float sum = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            float a = A[b * M * K + m * K + k];
            float bval = static_cast<float>(Bq[b * K * N + k * N + n]) * b_scale_batches[b][n];
            sum += a * bval;
          }
          CRef[b * M * N + m * N + n] = sum;
        }
      }
    }

    // Validate results
    for (size_t i = 0; i < M * N * BatchSize; ++i) {
      float abs_c_ref = std::abs(CRef[i]);
      float dynamic_rel_tol = (K <= 4) ? 0.05f : 0.03f;
      float rel_tol = dynamic_rel_tol * std::max(abs_c_ref, 1.0f);
      float abs_tol = 3.0f;
      float allowed = std::max(rel_tol, abs_tol);
      float diff = std::abs(C[i] - CRef[i]);
      ASSERT_LE(diff, allowed);
    }
  }

  static const char* GetTestSuiteName() {
    return "DynamicQgemm";
  }
};

class DynamicQgemmExecuteTest : public MlasTestFixture<MlasDynamicQgemmTest> {
 public:
  DynamicQgemmExecuteTest(size_t M, size_t N, size_t K, size_t BatchSize)
      : M_(M), N_(N), K_(K), BatchSize_(BatchSize) {}

  void TestBody() override {
    this->mlas_tester->Test(M_, N_, K_, BatchSize_);
  }
  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t BatchSize) {
    std::stringstream ss;
    ss << "M" << M << "_N" << N << "_K" << K << "_B" << BatchSize;

    std::string test_name = ss.str();

    testing::RegisterTest(
        MlasDynamicQgemmTest::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasDynamicQgemmTest>* {
          return new DynamicQgemmExecuteTest(M, N, K, BatchSize);
        });

    return 1;
  }

  static size_t RegisterAll(bool is_short_execute) {
    const std::vector<size_t> batch_size = is_short_execute ? std::vector<size_t>{1UL, 2UL, 4UL}
                                                            : std::vector<size_t>{1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL};
    size_t count = 0;
    const size_t sizes[] = {1, 4, 8, 16, 32, 64};
    for (size_t M : sizes)
      for (size_t N : sizes)
        for (size_t K : sizes)
          for (size_t B : batch_size)
            count += RegisterSingleTest(M, N, K, B);
    return count;
  }

 private:
  size_t M_, N_, K_, BatchSize_;
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return DynamicQgemmExecuteTest::RegisterAll(is_short_execute);
});
