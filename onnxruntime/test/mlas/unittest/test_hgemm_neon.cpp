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
  MLAS_FORCEINLINE void PackB(const MLAS_FP16* src, MLAS_FP16* dst) {
    size_t i = 0;
    for (; i + 16 <= N; i += 16) {
      for (size_t j = 0; j < K; ++j) {
        for (size_t k = 0; k < 16; ++k) {
          *dst = src[(i + k) * K + j];
          ++dst;
        }
      }
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
  MLAS_FORCEINLINE void Check(const MLAS_FP16* packed, const MLAS_FP16* ref) {
    size_t n = ((N + 7) & ~7) * K;
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(packed[i].val, ref[i].val) << " seed " << seed_ << " i " << i;
    }
  }

  template <size_t N, size_t K>
  void TestPackB() {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const auto* input = input_.GetFilledBuffer(N * K, InitializeBuffer);
    auto* packed = packed_.GetBuffer(K * ((N + 7) & ~7), true);
    auto* ref = ref_.GetBuffer(K * ((N + 7) & ~7), true);
    hgemm_neon::HPackB_TransposedB_Kernel(input, packed, N, K, K);
    PackB<N, K>(input, ref);
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
    TestPackB<1, 1>();
    TestPackB<1, 15>();
    TestPackB<1, 31>();
    TestPackB<8, 1>();
    TestPackB<8, 16>();
    TestPackB<9, 31>();
    TestPackB<9, 33>();
    TestPackB<15, 33>();
    TestPackB<17, 67>();
    TestPackB<17, 96>();
    TestPackB<265, 263>();
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
  MLAS_FORCEINLINE void HGemm(const MLAS_FP16* A, const MLAS_FP16* B, MLAS_FP16* C, MLAS_FP16 alpha, MLAS_FP16 beta) {
    float alphaf = alpha.ToFloat();
    float betaf = beta.ToFloat();
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float accu = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          accu += (A[m * K + k].ToFloat()) * (B[n * K + k].ToFloat());
        }
        C[m * N + n] = MLAS_FP16(accu * alphaf + C[m * N + n].ToFloat() * betaf);
      }
    }
  }

  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* C, const MLAS_FP16* ref) {
    size_t n = M * N;
    for (size_t i = 0; i < n; ++i) {
      ASSERT_TRUE(FloatEqual(C[i], ref[i], 0.02f, 0.055f))
          << " seed " << seed_ << " i " << i
          << " M " << M << " N " << N << " K " << K
          << " v0 " << C[i] << " v1 " << ref[i];
    }
  }

  template <size_t M, size_t K, size_t N>
  void TestHGemm(MLAS_FP16 alpha, MLAS_FP16 beta) {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const auto* A = A_.GetFilledBuffer(M * K, InitializeBuffer);
    const auto* B = B_.GetFilledBuffer(K * N, InitializeBuffer);
    auto* C = C_.GetBuffer(M * N, true);
    auto* ref = ref_.GetBuffer(M * N, true);
    hgemm_neon::HGemm_TransposedB_Kernel(A, B, C, M, N, K, K, K, N, alpha.val, beta.val);
    HGemm<M, K, N>(A, B, ref, alpha, beta);
    Check<M, K, N>(C, ref);
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

class MlasNeonHGemmTransposedPackedBTest : public MlasTestBase {
 private:
  std::random_device rd_;
  unsigned int seed_;
  std::mt19937 gen_;  // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> A_, B_, ref_, C_;

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void HGemm(const MLAS_FP16* A, const MLAS_FP16* B, MLAS_FP16* C, MLAS_FP16 alpha, MLAS_FP16 beta) {
    float alphaf = alpha.ToFloat();
    float betaf = beta.ToFloat();
    size_t n = 0;
    for (; n + 16 <= N; n += 16) {
      for (size_t i = 0; i < 16; ++i) {
        for (size_t m = 0; m < M; ++m) {
          float accu = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            accu += (A[m * K + k].ToFloat()) * (B[n * K + k * 16 + i].ToFloat());
          }
          C[m * N + n + i] = MLAS_FP16(accu * alphaf + C[m * N + n + i].ToFloat() * betaf);
        }
      }
    }
    if (n + 8 <= N) {
      for (size_t i = 0; i < 8; ++i) {
        for (size_t m = 0; m < M; ++m) {
          float accu = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            accu += (A[m * K + k].ToFloat()) * (B[n * K + k * 8 + i].ToFloat());
          }
          C[m * N + n + i] = MLAS_FP16(accu * alphaf + C[m * N + n + i].ToFloat() * betaf);
        }
      }
      n += 8;
    }
    if (n < N) {
      for (size_t i = 0; i < N - n; ++i) {
        for (size_t m = 0; m < M; ++m) {
          float accu = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            accu += (A[m * K + k].ToFloat()) * (B[n * K + k * 8 + i].ToFloat());
          }
          C[m * N + n + i] = MLAS_FP16(accu * alphaf + C[m * N + n + i].ToFloat() * betaf);
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
  MLAS_FORCEINLINE void Check(const MLAS_FP16* C, const MLAS_FP16* ref) {
    size_t n = M * N;
    for (size_t i = 0; i < n; ++i) {
      ASSERT_TRUE(FloatEqual(C[i], ref[i], 0.02f, 0.055f))
          << " seed " << seed_ << " i " << i
          << " M " << M << " K " << K << " N " << N
          << " v0 " << C[i] << " v1 " << ref[i];
    }
  }

  template <size_t M, size_t K, size_t N>
  void TestHGemm(MLAS_FP16 alpha, MLAS_FP16 beta) {
    auto InitializeBuffer = [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; i++) {
        buffer[i] = MLAS_FP16(distrib_(gen_));
      }
    };

    const auto* A = A_.GetFilledBuffer(M * K, InitializeBuffer);
    const auto* B = B_.GetFilledBuffer(K * ((N + 7) & ~7), InitializeBuffer);
    auto* C = C_.GetBuffer(M * N, true);
    auto* ref = ref_.GetBuffer(M * N, true);
    hgemm_neon::HGemm_TransposedPackedB_Kernel(A, B, C, M, N, K, K, N, alpha.val, beta.val);
    HGemm<M, K, N>(A, B, ref, alpha, beta);
    Check<M, K, N>(C, ref);
  }

 public:
  MlasNeonHGemmTransposedPackedBTest()
      : seed_(1928372), gen_(seed_), distrib_(-1.f, 1.f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonHGemmTransposedPackedB";
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

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void HGemm(const MLAS_FP16* A, const MLAS_FP16* B, MLAS_FP16* C, MLAS_FP16 alpha, MLAS_FP16 beta) {
    float alphaf = alpha.ToFloat();
    float betaf = beta.ToFloat();
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float accu = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          accu += (A[i * K + k].ToFloat()) * (B[j * K + k].ToFloat());
        }
        C[i * N + j] = MLAS_FP16(accu * alphaf + C[i * N + j].ToFloat() * betaf);
      }
    }
  }

  MLAS_FORCEINLINE
  bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
    float f0 = v0.ToFloat(), f1 = v1.ToFloat();
    return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
  }

  template <size_t M, size_t K, size_t N>
  MLAS_FORCEINLINE void Check(const MLAS_FP16* C, const MLAS_FP16* ref) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ASSERT_TRUE(FloatEqual(C[i * N + j], ref[i * N + j], 0.02f, 0.055f))
            << " seed " << seed_ << " i " << i << " j " << j
            << " M " << M << " K " << K << " N " << N
            << " v0 " << C[i * N + j] << " v1 " << ref[i * N + j];
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

    const auto* A = A_.GetFilledBuffer(M * K, InitializeBuffer);
    const auto* B = B_.GetFilledBuffer(K * N, InitializeBuffer);
    auto* C = C_.GetBuffer(M * N, true);
    auto* ref = ref_.GetBuffer(M * N, true);
    MlasGemm(CblasNoTrans, CblasTrans, M, N, K, A, K, B, K, C, N, alpha.val, beta.val, nullptr);
    HGemm<M, K, N>(A, B, ref, alpha, beta);
    Check<M, K, N>(C, ref);
  }

 public:
  MlasNeonHGemmTest()
      : seed_(192837), gen_(seed_), distrib_(-0.25f, 0.25f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonHGemm";
  }

  void ExecuteShort(void) override {
    TestHGemm<2, 1, 1>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<1, 128, 512>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 128, 513>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 128, 511>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<2, 129, 512>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<1, 127, 512>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<1, 513, 1023>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
    TestHGemm<2, 511, 1025>(MLAS_FP16(1.5f), MLAS_FP16(0.5f));
    TestHGemm<127, 513, 1023>(MLAS_FP16(1.0f), MLAS_FP16(0.0f));
    TestHGemm<129, 511, 1025>(MLAS_FP16(0.5f), MLAS_FP16(1.0f));
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasNeonHGemmPackBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonHGemmTransposedBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonHGemmTransposedPackedBTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonHGemmTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
