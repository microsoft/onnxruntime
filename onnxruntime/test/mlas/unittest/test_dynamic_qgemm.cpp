//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "mlas.h"
#include "test_util.h"
#include "core/mlas/inc/mlas.h"

#include <cmath>
#include <limits>

class MlasDynamicQgemmTestBase {
 private:
  MatrixGuardBuffer<float> buffer_a;
  MatrixGuardBuffer<float> buffer_bf;
  MatrixGuardBuffer<int8_t> buffer_bq;
  MatrixGuardBuffer<float> buffer_c;
  MatrixGuardBuffer<float> buffer_c_ref;

 protected:
  void Run(size_t M, size_t N, size_t K, size_t BatchSize,
           MLAS_THREADPOOL* threadpool, bool require_threadpool, const char* run_tag) {
    if (require_threadpool && threadpool == nullptr)
      GTEST_SKIP() << "Dynamic QGEMM threading path requested but no MLAS thread pool is available.";

    // The test harness assumes K>0 for generating/quantizing B (computes per-column min/max across K).
    // When K==0, the buffers are size 0 and the min/max logic dereferences invalid memory.
    if (K == 0) {
      GTEST_SKIP() << "Skipping DynamicQGEMM test with K==0: test harness assumes K>0 for quantization setup.";
    }

    // Setup buffers for holding various data
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
    std::vector<std::vector<int8_t>> a_quant_batches(BatchSize, std::vector<int8_t>(M * K));
    std::vector<std::vector<float>> a_scale_batches(BatchSize, std::vector<float>(M));
    std::vector<std::vector<int32_t>> a_zero_point_batches(BatchSize, std::vector<int32_t>(M));

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

    // Quantize A rows to match the dynamic quantization performed by the kernel.
    for (size_t b = 0; b < BatchSize; ++b) {
      for (size_t m = 0; m < M; ++m) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (size_t k = 0; k < K; ++k) {
          float v = A[b * M * K + m * K + k];
          min_val = std::min(min_val, v);
          max_val = std::max(max_val, v);
        }
        float rmin = std::min(0.0f, min_val);
        float rmax = std::max(0.0f, max_val);
        float inv_scale = (rmax == rmin) ? 1.0f : 255.0f / (rmax - rmin);
        float scale = inv_scale ? 1.0f / inv_scale : 0.0f;
        float descaled_min = rmin * inv_scale;
        float descaled_max = rmax * inv_scale;
        float zero_point_from_min_error = -128.0f + descaled_min;
        float zero_point_from_max_error = 127.0f + descaled_max;
        float zero_point = (zero_point_from_min_error + zero_point_from_max_error > 0.0f)
                               ? (-128.0f - descaled_min)
                               : (127.0f - descaled_max);
        zero_point = std::clamp(zero_point, -128.0f, 127.0f);
        int32_t zp = static_cast<int32_t>(std::nearbyint(zero_point));

        a_scale_batches[b][m] = scale;
        a_zero_point_batches[b][m] = zp;

        for (size_t k = 0; k < K; ++k) {
          float v = A[b * M * K + m * K + k];
          int32_t q = static_cast<int32_t>(std::round(v * inv_scale)) + zp;
          q = std::clamp(q, -128, 127);
          a_quant_batches[b][m * K + k] = static_cast<int8_t>(q);
        }
      }
    }

    // Prepare kernel parameters
    MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS shape{M, N, K};

    const size_t packed_b_stride = MlasDynamicQgemmPackBSize(N, K);
    std::vector<uint8_t> packed_b_storage(BatchSize * packed_b_stride);
    std::vector<MLAS_GEMM_DYN_QUANT_DATA_PARAMS> params(BatchSize);

    for (size_t b = 0; b < BatchSize; ++b) {
      params[b].A = A + b * M * K;
      params[b].lda = K;
      params[b].C = C + b * M * N;
      params[b].ldc = N;

      // Pack b matrix using MlasDynamicQgemmPackBSize & MlasDynamicQgemmPackB.
      // When packed_b_stride is 0 (e.g., degenerate shapes like K==0), avoid taking data()
      // from a zero-sized vector as that may be null/invalid on some platforms.
      void* packed_b = packed_b_stride == 0 ? nullptr : (packed_b_storage.data() + b * packed_b_stride);

      if (packed_b_stride != 0) {
        MlasDynamicQgemmPackB(N, K,
                              Bq + b * K * N,
                              b_scale_batches[b].data(),
                              b_bias_batches[b].data(),
                              packed_b);
      }

      params[b].PackedB = packed_b;
    }

    // Compute reference result
    for (size_t b = 0; b < BatchSize; ++b) {
      for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
          float sum = 0.0f;
          const float a_scale = a_scale_batches[b][m];
          const int32_t a_zero_point = a_zero_point_batches[b][m];
          for (size_t k = 0; k < K; ++k) {
            int32_t a_q = static_cast<int32_t>(a_quant_batches[b][m * K + k]);
            float a = static_cast<float>(a_q - a_zero_point) * a_scale;
            float bval = static_cast<float>(Bq[b * K * N + k * N + n]) * b_scale_batches[b][n];
            sum += a * bval;
          }
          CRef[b * M * N + m * N + n] = sum;
        }
      }
    }

    std::fill(C, C + M * N * BatchSize, 0.0f);
    MlasDynamicQGemmBatch(shape, params.data(), BatchSize, threadpool);

    // Validate results
    auto validate = [&](const char* tag) {
      SCOPED_TRACE(tag);

      for (size_t i = 0; i < M * N * BatchSize; ++i) {
        float abs_tol = 0.001f;
        float diff = std::abs(C[i] - CRef[i]);
        ASSERT_LE(diff, abs_tol);
      }
    };

    validate(run_tag);
  }
};

class MlasDynamicQgemmSingleThreadTest : public MlasDynamicQgemmTestBase {
 public:
  void Test(size_t M, size_t N, size_t K, size_t BatchSize) {
    // Currently, MlasDynamicQGemmBatch() and associated functions require SME or else they are no-ops.
    if (!MlasIsDynamicQGemmAvailable())
      GTEST_SKIP() << "MlasDynamicQGemmBatch() requires ARM64 SME or SME2 but it was not detected. Skipping test.";
    Run(M, N, K, BatchSize, /*threadpool*/ nullptr, /*require_threadpool*/ false, "SingleThread");
  }
  static const char* GetTestSuiteName() { return "DynamicQgemmSingleThread"; }
};

class MlasDynamicQgemmThreadPoolTest : public MlasDynamicQgemmTestBase {
 public:
  void Test(size_t M, size_t N, size_t K, size_t BatchSize) {
    // Currently, MlasDynamicQGemmBatch() and associated functions require SME or else they are no-ops.
    if (!MlasIsDynamicQGemmAvailable())
      GTEST_SKIP() << "MlasDynamicQGemmBatch() requires ARM64 SME or SME2 but it was not detected. Skipping test.";
    MLAS_THREADPOOL* tp = GetMlasThreadPool();
    if (!tp) GTEST_SKIP() << "Mlas thread pool not available";
    Run(M, N, K, BatchSize, tp, /*require_threadpool*/ true, "ThreadPool");
  }
  static const char* GetTestSuiteName() { return "DynamicQgemmThreaded"; }
};

template <typename TMlasTester>
class DynamicQgemmExecuteTest : public MlasTestFixture<TMlasTester> {
 public:
  DynamicQgemmExecuteTest(size_t M, size_t N, size_t K, size_t BatchSize)
      : M_(M), N_(N), K_(K), BatchSize_(BatchSize) {}

  void TestBody() override {
    MlasTestFixture<TMlasTester>::mlas_tester->Test(M_, N_, K_, BatchSize_);
  }
  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t BatchSize) {
    std::stringstream ss;
    ss << "M" << M << "_N" << N << "_K" << K << "_B" << BatchSize;
    std::string test_name = ss.str();

    testing::RegisterTest(
        TMlasTester::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<TMlasTester>* {
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

    // Zero-dimension probes: these should exercise the early-return behavior in implementations
    // that treat M==0 or N==0 as a no-op.
    count += RegisterSingleTest(0, 16, 16, 1);  // M==0
    count += RegisterSingleTest(16, 0, 16, 1);  // N==0

    // K==0 probe: included to observe behavior when the reduction dimension is zero.
    count += RegisterSingleTest(16, 16, 0, 1);  // K==0

    return count;
  }

 private:
  size_t M_, N_, K_, BatchSize_;
};

static UNUSED_VARIABLE bool added_single = AddTestRegister([](bool is_short_execute) {
  return DynamicQgemmExecuteTest<MlasDynamicQgemmSingleThreadTest>::RegisterAll(is_short_execute);
});

static UNUSED_VARIABLE bool added_threaded = AddTestRegister([](bool is_short_execute) {
  return DynamicQgemmExecuteTest<MlasDynamicQgemmThreadPoolTest>::RegisterAll(is_short_execute);
});
