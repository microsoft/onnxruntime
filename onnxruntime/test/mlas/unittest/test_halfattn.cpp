// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/halfattn.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

//
// Tests MlasHalfAttnQK and MlasHalfAttnPV, and on ARM64 each vector kernel
// directly, against fp64 references on the same fp16 inputs. The tolerances
// allow fp32 rounding but not fp16 accumulation, which is off by orders of
// magnitude more on the long sums.
//
class MlasHalfAttnTestBase : public MlasTestBase {
 protected:
  using QKFn = MLAS_HALF_ATTN_DISPATCH::QK_Fn;
  using PVFn = MLAS_HALF_ATTN_DISPATCH::PV_Fn;

  static void ApiQK(size_t M, size_t N, size_t K, float Alpha, const MLAS_FP16* A, size_t lda, const MLAS_FP16* B,
                    size_t ldb, float* C, size_t ldc) {
    MlasHalfAttnQK(M, N, K, Alpha, A, lda, B, ldb, C, ldc);
  }

  static void ApiPV(size_t M, size_t N, size_t K, const float* A, size_t lda, const MLAS_FP16* B, size_t ldb,
                    float* C, size_t ldc) {
    MlasHalfAttnPV(M, N, K, A, lda, B, ldb, C, ldc);
  }

 private:
  MatrixGuardBuffer<MLAS_FP16> BufferA16;
  MatrixGuardBuffer<float> BufferA32;
  MatrixGuardBuffer<MLAS_FP16> BufferB;
  MatrixGuardBuffer<float> BufferC;
  std::mt19937 rng_{1234};

  void FillHalf(MLAS_FP16* data, size_t count, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < count; ++i) {
      data[i] = MLAS_FP16(dist(rng_));
    }
  }

  void FillFloat(float* data, size_t count, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < count; ++i) {
      data[i] = dist(rng_);
    }
  }

  // Error allowed for an fp32 sum of `terms` random terms whose magnitudes add up to `magnitude`.
  static double SumTolerance(size_t terms, double magnitude) {
    return (4.0 * std::sqrt(static_cast<double>(terms)) + 8.0) * std::ldexp(1.0, -24) * magnitude + 1e-30;
  }

  void TestQK(QKFn* qk, const char* name, size_t M, size_t N, size_t K) {
    const size_t lda = K + 3;
    const size_t ldb = K + 5;
    const size_t ldc = N + 2;
    MLAS_FP16* A = BufferA16.GetBuffer(M * lda);
    MLAS_FP16* B = BufferB.GetBuffer(N * ldb);
    float* C = BufferC.GetBuffer(M * ldc);
    FillHalf(A, M * lda, -2.0f, 2.0f);
    FillHalf(B, N * ldb, -2.0f, 2.0f);
    const float sentinel = -12345.0f;
    std::fill_n(C, M * ldc, sentinel);
    const float alpha = K > 0 ? 1.0f / std::sqrt(static_cast<float>(K)) : 0.5f;

    qk(M, N, K, alpha, A, lda, B, ldb, C, ldc);

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        double sum = 0.0;
        double magnitude = 0.0;
        for (size_t k = 0; k < K; ++k) {
          const double product = static_cast<double>(A[m * lda + k].ToFloat()) * B[n * ldb + k].ToFloat();
          sum += product;
          magnitude += std::fabs(product);
        }
        const double expected = alpha * sum;
        const double actual = C[m * ldc + n];
        ASSERT_LE(std::fabs(actual - expected), SumTolerance(K, alpha * magnitude))
            << name << " QK M=" << M << " N=" << N << " K=" << K << " @ (" << m << ", " << n << ")";
      }
      for (size_t n = N; n < ldc; ++n) {
        ASSERT_EQ(C[m * ldc + n], sentinel) << name << " QK wrote past N, M=" << M << " N=" << N << " K=" << K;
      }
    }
  }

  void TestPV(PVFn* pv, const char* name, size_t M, size_t N, size_t K) {
    const size_t lda = K + 1;
    const size_t ldb = N + 6;
    const size_t ldc = N + 3;
    float* A = BufferA32.GetBuffer(M * lda);
    MLAS_FP16* B = BufferB.GetBuffer(K * ldb);
    float* C = BufferC.GetBuffer(M * ldc);
    FillFloat(A, M * lda, 0.0f, 1.0f);
    FillHalf(B, K * ldb, -2.0f, 2.0f);
    FillFloat(C, M * ldc, -1.0f, 1.0f);
    std::vector<float> initial(C, C + M * ldc);

    pv(M, N, K, A, lda, B, ldb, C, ldc);

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        double sum = initial[m * ldc + n];
        double magnitude = std::fabs(sum);
        for (size_t k = 0; k < K; ++k) {
          const double product = static_cast<double>(A[m * lda + k]) * B[k * ldb + n].ToFloat();
          sum += product;
          magnitude += std::fabs(product);
        }
        ASSERT_LE(std::fabs(C[m * ldc + n] - sum), SumTolerance(K, magnitude))
            << name << " PV M=" << M << " N=" << N << " K=" << K << " @ (" << m << ", " << n << ")";
      }
      for (size_t n = N; n < ldc; ++n) {
        ASSERT_EQ(C[m * ldc + n], initial[m * ldc + n])
            << name << " PV wrote past N, M=" << M << " N=" << N << " K=" << K;
      }
    }
  }

 protected:
  void TestKernels(QKFn* qk, PVFn* pv, const char* name) {
    static constexpr size_t kRows[] = {1, 2, 3, 4, 5, 8, 12, 16};

    // QK: N cache rows of head size K. Head sizes cover the 8-wide main loop, its tail and the column chunks of the
    // widen kernel (256).
    static constexpr size_t kQKCacheRows[] = {1, 3, 4, 7, 64, 128};
    static constexpr size_t kQKHeadSizes[] = {0, 1, 7, 8, 64, 72, 128, 136, 256, 300, 520};
    for (size_t M : kRows) {
      for (size_t N : kQKCacheRows) {
        for (size_t K : kQKHeadSizes) {
          TestQK(qk, name, M, N, K);
        }
      }
    }

    // PV: K cache rows of head size N. Head sizes cover the 16, 8 and 4 column blocks and the scalar columns.
    static constexpr size_t kPVHeadSizes[] = {1, 3, 4, 8, 12, 16, 20, 27, 64, 80, 128, 136};
    static constexpr size_t kPVCacheRows[] = {1, 3, 4, 5, 64, 127, 128};
    for (size_t M : kRows) {
      for (size_t N : kPVHeadSizes) {
        for (size_t K : kPVCacheRows) {
          TestPV(pv, name, M, N, K);
        }
      }
    }

    // Long sums, where fp16 accumulation would be off by far more than the tolerance.
    TestQK(qk, name, 3, 16, 4096);
    TestPV(pv, name, 3, 128, 8192);
  }
};

class MlasHalfAttnTest : public MlasHalfAttnTestBase {
 public:
  static const char* GetTestSuiteName() { return "HalfAttn"; }
  void ExecuteShort(void) override { TestKernels(ApiQK, ApiPV, "MlasHalfAttn"); }
};

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

class MlasHalfAttnNeonTest : public MlasHalfAttnTestBase {
 public:
  static const char* GetTestSuiteName() { return "HalfAttnNeon"; }
  void ExecuteShort(void) override {
    TestKernels(MlasHalfAttnDispatchNeon.QK, MlasHalfAttnDispatchNeon.PV, "Neon");
  }
};

class MlasHalfAttnNeonFhmTest : public MlasHalfAttnTestBase {
 public:
  static const char* GetTestSuiteName() { return "HalfAttnNeonFhm"; }
  void ExecuteShort(void) override {
    TestKernels(MlasHalfAttnDispatchNeonFhm.QK, MlasHalfAttnDispatchNeonFhm.PV, "NeonFhm");
  }
};

#endif

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasHalfAttnTest>::RegisterShortExecute();
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    // The vector kernels run only on CPUs with fp16 arithmetic, and the FHM one only with FEAT_FHM.
    if (MlasFp16AccelerationSupported()) {
      count += MlasDirectShortExecuteTests<MlasHalfAttnNeonTest>::RegisterShortExecute();
      if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_FHM()) {
        count += MlasDirectShortExecuteTests<MlasHalfAttnNeonFhmTest>::RegisterShortExecute();
      }
    }
#endif
  }
  return count;
});
