/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_hgemm_neon.cpp

Abstract:

    Tests for MLAS fp16 GEMM on ARM CPU.

--*/

#include <vector>
#include <random>

#include "test/mlas/unittest/test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/halfgemm.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

class MlasNeonHGemmPackBTest : public MlasTestBase {
 private:
  std::random_device rd_;
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> input_, ref_, packed_;

  template <size_t N, size_t K>
  MLAS_FORCEINLINE void PackB_TransposedB(const MLAS_FP16* src, MLAS_FP16* dst) {
    size_t i = 0;
    for (; i + 32 <= N; i += 32) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < 32; ++k) {
          *dst = src[(i + k) * K + j];
          ++dst;
        }
      }
    }
    if (i + 16 <= N) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < 16; ++k) {
          *dst = src[(i + k) * K + j];
          ++dst;
        }
      }
      i += 16;
    }
    if (i + 8 <= N) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < 8; ++k) {
          *dst = src[(i + k) * K + j];
          ++dst;
        }
      }
      i += 8;
    }
    if (i < N) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < N - i; ++k) {
          *dst = src[(i + k) * K + j];
          ++dst;
        }
        dst += 8 - (N - i);
      }
    }
  }

  template <size_t N, size_t K>
  MLAS_FORCEINLINE void PackB_B(const MLAS_FP16* src, MLAS_FP16* dst) {
    size_t i = 0;
    for (; i + 32 <= N; i += 32) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < 32; ++k) {
          *dst = src[(i + k) + j * N];
          ++dst;
        }
      }
    }
    for (; i + 16 <= N; i += 16) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < 16; ++k) {
          *dst = src[(i + k) + j * N];
          ++dst;
        }
      }
    }
    if (i + 8 <= N) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < 8; ++k) {
          *dst = src[(i + k) + j * N];
          ++dst;
        }
      }
      i += 8;
    }
    if (i < N) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < N - i; ++k) {
          *dst = src[(i + k) + j * N];
          ++dst;
        }
        dst += 8 - (N - i);
      }
    }
  }

  template <size_t N, size_t K>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* packed, const MLAS_FP16* ref) {
    size_t j = 0;
    for (; j + 31 < N; j += 32) {
      for (size_t i = 0; i < 32 * K; ++i) {
        ASSERT_EQ(packed[j * K + i].val, ref[j * K + i].val)
            << " seed " << seed_ << " K " << i / 32 << " N " << j + i % 32;
      }
    }
    for (; j + 15 < N; j += 16) {
      for (size_t i = 0; i < 16 * K; ++i) {
        ASSERT_EQ(packed[j * K + i].val, ref[j * K + i].val)
            << " seed " << seed_ << " K " << i / 16 << " N " << j + i % 16;
      }
    }
    if (j + 7 < N) {
      for (size_t i = 0; i < 8 * K; ++i) {
        ASSERT_EQ(packed[j * K + i].val, ref[j * K + i].val)
            << " seed " << seed_ << " K " << i / 8 << " N " << j + i % 8;
      }
      j += 8;
    }
    if (j < N) {
      for (size_t i = 0; i < K; ++i) {
        for (size_t k = 0; k < N - j; ++k) {
          ASSERT_EQ(packed[j * K + i * 8 + k].val, ref[j * K + i * 8 + k].val)
              << " seed " << seed_ << " K " << i << " N " << j + k;
        }
      }
    }
  }

  template <size_t N, size_t K>
  void TestPackB_TransposedB() {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const auto* input = input_.GetFilledBuffer(N * K, InitializeBuffer);
    auto* packed = packed_.GetBuffer(K * ((N + 7) & ~7), true);
    auto* ref = ref_.GetBuffer(K * ((N + 7) & ~7), true);
    hgemm_neon::HPackB_TransposedB_Kernel(input, packed, N, K, K);
    PackB_TransposedB<N, K>(input, ref);
    Check<N, K>(packed, ref);
  }

  template <size_t N, size_t K>
  void TestPackB_B() {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const auto* input = input_.GetFilledBuffer(N * K, InitializeBuffer);
    auto* packed = packed_.GetBuffer(K * ((N + 7) & ~7), true);
    auto* ref = ref_.GetBuffer(K * ((N + 7) & ~7), true);
    hgemm_neon::HPackB_B_Kernel(input, packed, N, K, N);
    PackB_B<N, K>(input, ref);
    Check<N, K>(packed, ref);
  }

 public:
  MlasNeonHGemmPackBTest()
      : seed_(rd_()), gen_(seed_), distrib_(-100.f, 100.f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonHGemmPackB";
  }

  void ExecuteShort(void) override {
    TestPackB_TransposedB<1, 1>();
    TestPackB_TransposedB<1, 15>();
    TestPackB_TransposedB<1, 31>();
    TestPackB_TransposedB<8, 1>();
    TestPackB_TransposedB<8, 16>();
    TestPackB_TransposedB<9, 31>();
    TestPackB_TransposedB<9, 33>();
    TestPackB_TransposedB<31, 33>();
    TestPackB_TransposedB<33, 67>();
    TestPackB_TransposedB<63, 96>();
    TestPackB_TransposedB<271, 263>();
    TestPackB_B<1, 1>();
    TestPackB_B<1, 15>();
    TestPackB_B<1, 31>();
    TestPackB_B<8, 1>();
    TestPackB_B<8, 16>();
    TestPackB_B<9, 31>();
    TestPackB_B<9, 33>();
    TestPackB_B<31, 31>();
    TestPackB_B<63, 33>();
    TestPackB_B<33, 31>();
    TestPackB_B<33, 33>();
    TestPackB_B<65, 67>();
    TestPackB_B<65, 96>();
    TestPackB_B<271, 263>();
  }
};

class MlasNeonHGemmTransposedBTest : public MlasTestBase {
 private:
  std::random_device rd_;
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> A_, B_, ref_, C_;

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void HGemm(
      const MLAS_FP16* A, const MLAS_FP16* B, MLAS_FP16* C, MLAS_FP16 alpha, MLAS_FP16 beta,
      size_t lda, size_t ldb, size_t ldc) {
    float alphaf = alpha.ToFloat();
    float betaf = beta.ToFloat();
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float accu = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          accu += (A[m * lda + k].ToFloat()) * (B[n * ldb + k].ToFloat());
        }
        C[m * ldc + n] = MLAS_FP16(accu * alphaf + C[m * ldc + n].ToFloat() * betaf);
      }
    }
  }

  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t M, size_t N>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* C, const MLAS_FP16* ref, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t k = i * ldc + j;
        ASSERT_TRUE(FloatEqual(C[k], ref[k], 0.02f, 0.055f))
            << " seed " << seed_ << " i " << i << " j " << j
            << " M " << M << " N " << N << " ldc " << ldc
            << " value " << C[k] << " ref " << ref[k];
      }
    }
  }

  template <size_t M, size_t N>
  MLAS_FORCEINLINE void Copy(const MLAS_FP16* C, MLAS_FP16* ref, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t k = i * ldc + j;
        ref[k] = C[k];
      }
    }
  }

  template <size_t M, size_t K, size_t N>
  void TestHGemm(MLAS_FP16 alpha, MLAS_FP16 beta) {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const size_t lda = ((K + 7) & ~7);
    const size_t ldb = ((K + 7) & ~7) + 8;
    const size_t ldc = ((N + 7) & ~7);
    const auto* A = A_.GetFilledBuffer(M * lda, InitializeBuffer);
    const auto* B = B_.GetFilledBuffer(ldb * N, InitializeBuffer);
    auto* C = C_.GetFilledBuffer(M * ldc, InitializeBuffer);
    auto* ref = ref_.GetBuffer(M * ldc, true);
    Copy<M, N>(C, ref, ldc);
    hgemm_neon::HGemm_TransposedB_Kernel(A, B, C, M, N, K, lda, ldb, ldc, alpha.val, beta.val);
    HGemm<M, K, N>(A, B, ref, alpha, beta, lda, ldb, ldc);
    Check<M, N>(C, ref, ldc);
  }

 public:
  MlasNeonHGemmTransposedBTest()
      : seed_(1928375), gen_(seed_), distrib_(-1.f, 1.f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonHGemmTransposedB";
  }

  void ExecuteShort(void) override {
    TestHGemm<2, 1, 1>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<1, 1, 1>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 1, 1>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 15, 17>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 17, 15>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 17, 15>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 33, 31>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 31, 32>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 32, 33>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 78, 263>(MLAS_FP16(0.5f), MLAS_FP16(0.0f));
    TestHGemm<2, 267, 79>(MLAS_FP16(1.5f), MLAS_FP16(1.0f));
  }
};

class MlasNeonHGemmBTest : public MlasTestBase {
 private:
  std::random_device rd_;
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> A_, B_, ref_, C_;

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void HGemm(
      const MLAS_FP16* A, const MLAS_FP16* B, MLAS_FP16* C, MLAS_FP16 alpha, MLAS_FP16 beta,
      size_t lda, size_t ldb, size_t ldc) {
    float alphaf = alpha.ToFloat();
    float betaf = beta.ToFloat();
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float accu = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          accu += (A[m * lda + k].ToFloat()) * (B[n + k * ldb].ToFloat());
        }
        C[m * ldc + n] = MLAS_FP16(accu * alphaf + C[m * ldc + n].ToFloat() * betaf);
      }
    }
  }

  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t M, size_t N>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* C, const MLAS_FP16* ref, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t idx = i * ldc + j;
        ASSERT_TRUE(FloatEqual(C[idx], ref[idx], 0.02f, 0.055f))
            << " seed " << seed_ << " i " << i << " j " << j
            << " M " << M << " N " << N << " ldc " << ldc
            << " value " << C[idx] << " ref " << ref[idx];
      }
    }
  }

  template <size_t M, size_t N>
  MLAS_FORCEINLINE void Copy(const MLAS_FP16* C, MLAS_FP16* ref, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t idx = i * ldc + j;
        ref[idx] = C[idx];
      }
    }
  }

  template <size_t M, size_t K, size_t N>
  void TestHGemm(MLAS_FP16 alpha, MLAS_FP16 beta) {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const size_t lda = ((K + 7) & ~7);
    const size_t ldb = ((N + 7) & ~7) + 8;
    const size_t ldc = ((N + 7) & ~7);
    const auto* A = A_.GetFilledBuffer(M * lda, InitializeBuffer);
    const auto* B = B_.GetFilledBuffer(K * ldb, InitializeBuffer);
    auto* C = C_.GetFilledBuffer(M * ldc, InitializeBuffer);
    auto* ref = ref_.GetBuffer(M * ldc, true);
    Copy<M, N>(C, ref, ldc);
    hgemm_neon::HGemm_B_Kernel(A, B, C, M, N, K, lda, ldb, ldc, alpha.val, beta.val);
    HGemm<M, K, N>(A, B, ref, alpha, beta, lda, ldb, ldc);
    Check<M, N>(C, ref, ldc);
  }

 public:
  MlasNeonHGemmBTest()
      : seed_(172387), gen_(seed_), distrib_(-1.f, 1.f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonHGemmB";
  }

  void ExecuteShort(void) override {
    TestHGemm<2, 1, 1>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<1, 1, 1>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 1, 1>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 15, 17>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 17, 15>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 17, 15>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 33, 31>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 31, 32>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 32, 33>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 78, 263>(MLAS_FP16(0.5f), MLAS_FP16(0.0f));
    TestHGemm<2, 267, 79>(MLAS_FP16(1.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 1, 1>(MLAS_FP16(1.0f), MLAS_FP16(1.0f));
    TestHGemm<1, 1, 1>(MLAS_FP16(1.f), MLAS_FP16(0.0f));
    TestHGemm<2, 1, 1>(MLAS_FP16(1.f), MLAS_FP16(0.f));
    TestHGemm<1, 15, 17>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 17, 15>(MLAS_FP16(1.f), MLAS_FP16(1.0f));
    TestHGemm<1, 17, 15>(MLAS_FP16(1.f), MLAS_FP16(1.f));
    TestHGemm<1, 33, 31>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 31, 32>(MLAS_FP16(1.f), MLAS_FP16(1.0f));
    TestHGemm<1, 32, 33>(MLAS_FP16(1.f), MLAS_FP16(0.f));
    TestHGemm<1, 78, 263>(MLAS_FP16(1.f), MLAS_FP16(0.0f));
    TestHGemm<2, 267, 79>(MLAS_FP16(1.f), MLAS_FP16(1.0f));
    TestHGemm<2, 65, 65>(MLAS_FP16(1.f), MLAS_FP16(0.0f));
    TestHGemm<2, 63, 63>(MLAS_FP16(1.f), MLAS_FP16(0.0f));
    TestHGemm<2, 65, 63>(MLAS_FP16(1.f), MLAS_FP16(0.0f));
    TestHGemm<2, 63, 65>(MLAS_FP16(1.f), MLAS_FP16(0.0f));
  }
};

class MlasNeonHGemmPackedBTest : public MlasTestBase {
 private:
  std::random_device rd_;
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> A_, B_, ref_, C_;

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void HGemm(
      const MLAS_FP16* A, const MLAS_FP16* B, MLAS_FP16* C, MLAS_FP16 alpha, MLAS_FP16 beta,
      size_t lda, size_t ldc) {
    float alphaf = alpha.ToFloat();
    float betaf = beta.ToFloat();
    size_t n = 0;
    for (; n + 32 <= N; n += 32) {
      for (size_t i = 0; i < 32; ++i) {
        for (size_t m = 0; m < M; ++m) {
          float accu = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            accu += (A[m * lda + k].ToFloat()) * (B[n * K + k * 32 + i].ToFloat());
          }
          C[m * ldc + n + i] = MLAS_FP16(accu * alphaf + C[m * ldc + n + i].ToFloat() * betaf);
        }
      }
    }
    for (; n + 16 <= N; n += 16) {
      for (size_t i = 0; i < 16; ++i) {
        for (size_t m = 0; m < M; ++m) {
          float accu = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            accu += (A[m * lda + k].ToFloat()) * (B[n * K + k * 16 + i].ToFloat());
          }
          C[m * ldc + n + i] = MLAS_FP16(accu * alphaf + C[m * ldc + n + i].ToFloat() * betaf);
        }
      }
    }
    if (n + 8 <= N) {
      for (size_t i = 0; i < 8; ++i) {
        for (size_t m = 0; m < M; ++m) {
          float accu = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            accu += (A[m * lda + k].ToFloat()) * (B[n * K + k * 8 + i].ToFloat());
          }
          C[m * ldc + n + i] = MLAS_FP16(accu * alphaf + C[m * ldc + n + i].ToFloat() * betaf);
        }
      }
      n += 8;
    }
    if (n < N) {
      for (size_t i = 0; i < N - n; ++i) {
        for (size_t m = 0; m < M; ++m) {
          float accu = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            accu += (A[m * lda + k].ToFloat()) * (B[n * K + k * 8 + i].ToFloat());
          }
          C[m * ldc + n + i] = MLAS_FP16(accu * alphaf + C[m * ldc + n + i].ToFloat() * betaf);
        }
      }
    }
  }

  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* C, const MLAS_FP16* ref, const size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t k = i * ldc + j;
        ASSERT_TRUE(FloatEqual(C[k], ref[k], 0.02f, 0.055f))
            << " seed " << seed_ << " i " << i << " j " << j
            << " M " << M << " K " << K << " N " << N
            << " value " << C[k] << " ref " << ref[k];
      }
    }
  }

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void Copy(const MLAS_FP16* C, MLAS_FP16* ref, const size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t k = i * ldc + j;
        ref[k] = C[k];
      }
    }
  }

  template <size_t M, size_t K, size_t N>
  void TestHGemm(MLAS_FP16 alpha, MLAS_FP16 beta) {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const size_t lda = ((K + 7) & ~7) + 8;
    const size_t ldc = ((N + 7) & ~7);
    const auto* A = A_.GetFilledBuffer(M * lda, InitializeBuffer);
    const auto* B = B_.GetFilledBuffer(K * ((N + 7) & ~7), InitializeBuffer);
    auto* C = C_.GetFilledBuffer(M * ldc, InitializeBuffer);
    auto* ref = ref_.GetBuffer(M * ldc, true);
    Copy<M, K, N>(C, ref, ldc);
    hgemm_neon::HGemm_PackedB_Kernel(A, B, C, M, N, K, lda, ldc, alpha.val, beta.val);
    HGemm<M, K, N>(A, B, ref, alpha, beta, lda, ldc);
    Check<M, K, N>(C, ref, ldc);
  }

 public:
  MlasNeonHGemmPackedBTest()
      : seed_(1928372), gen_(), distrib_(-1.f, 1.f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonHGemmPackedB";
  }

  void ExecuteShort(void) override {
    TestHGemm<2, 1, 1>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<1, 1, 1>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 1, 1>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 15, 17>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 17, 15>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 17, 15>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 33, 31>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 31, 32>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 32, 33>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 78, 263>(MLAS_FP16(0.5f), MLAS_FP16(0.0f));
    TestHGemm<2, 267, 79>(MLAS_FP16(1.5f), MLAS_FP16(1.0f));
  }
};

class MlasNeonHGemmTest : public MlasTestBase {
 private:
  std::random_device rd_;
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> A_, B_, ref_, C_;

  template <size_t M, size_t K, size_t N, bool transA, bool transB>
  MLAS_FORCEINLINE void HGemm(const MLAS_FP16* A, const MLAS_FP16* B, MLAS_FP16* C, MLAS_FP16 alpha, MLAS_FP16 beta,
                              size_t lda, size_t ldb, size_t ldc) {
    float alphaf = alpha.ToFloat();
    float betaf = beta.ToFloat();
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float accu = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          accu += (A[transA ? k * lda + i : i * lda + k].ToFloat()) * (B[transB ? j * ldb + k : k * ldb + j].ToFloat());
        }
        C[i * ldc + j] = MLAS_FP16(accu * alphaf + C[i * ldc + j].ToFloat() * betaf);
      }
    }
  }

  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* C, const MLAS_FP16* ref, const size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ASSERT_TRUE(FloatEqual(C[i * ldc + j], ref[i * ldc + j], 0.02f, 0.055f))
            << " seed " << seed_ << " i " << i << " j " << j
            << " M " << M << " K " << K << " N " << N
            << " value " << C[i * ldc + j] << " ref " << ref[i * ldc + j];
      }
    }
  }

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void Copy(const MLAS_FP16* C, MLAS_FP16* ref, const size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ref[i * ldc + j] = C[i * ldc + j];
      }
    }
  }

  template <size_t M, size_t K, size_t N, bool transA, bool transB>
  void TestHGemm(MLAS_FP16 alpha, MLAS_FP16 beta) {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const size_t lda = transA ? (M + 15) & (~15) : (K + 15) & (~15);
    const size_t ldb = transB ? (K + 7) & (~7) : (N + 15) & (~15);
    const size_t ldc = (N + 7) & (~7);
    const auto* A = A_.GetFilledBuffer(transA ? lda * K : M * lda, InitializeBuffer);
    const auto* B = B_.GetFilledBuffer(transB ? ldb * N : K * ldb, InitializeBuffer);
    auto* C = C_.GetFilledBuffer(M * ldc, InitializeBuffer);
    auto* ref = ref_.GetBuffer(M * ldc, true);
    Copy<M, K, N>(C, ref, ldc);
    MlasGemm(transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
             M, N, K, A, lda, B, ldb, C, ldc, alpha.val, beta.val, nullptr);
    HGemm<M, K, N, transA, transB>(A, B, ref, alpha, beta, lda, ldb, ldc);
    Check<M, K, N>(C, ref, ldc);
  }

 public:
  MlasNeonHGemmTest()
      : seed_(192837), gen_(seed_), distrib_(-0.25f, 0.25f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonHGemm";
  }

  // TODO(fajin): test beta
  void ExecuteShort(void) override {
    TestHGemm<2, 1, 1, false, true>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<1, 128, 512, false, true>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 128, 513, false, true>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 128, 511, false, true>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 129, 512, false, true>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 127, 512, false, true>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 513, 1023, false, true>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 511, 1025, false, true>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<127, 513, 1023, false, true>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<129, 511, 1025, false, true>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 1, 1, false, false>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<1, 128, 512, false, false>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 128, 513, false, false>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 128, 511, false, false>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 129, 512, false, false>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 127, 512, false, false>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 513, 1023, false, false>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 511, 1025, false, false>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<127, 513, 1023, false, false>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<129, 511, 1025, false, false>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<129, 513, 1025, false, false>(MLAS_FP16(0.5f), MLAS_FP16(0.5f));
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasNeonHGemmPackBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonHGemmTransposedBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonHGemmBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonHGemmPackedBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonHGemmTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
