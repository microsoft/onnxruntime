/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    test_halfgemm_sve.cpp

Abstract:

    Tests for the SVE FP16 GEMM kernels and the HGEMM driver.

--*/

#include "test_util.h"
#include "test_fp16.h"

#include "core/mlas/lib/mlasi.h"

#if defined(MLAS_USE_SVE) && defined(MLAS_TARGET_ARM64)

#include <sstream>
#include <vector>

#include "core/mlas/lib/sve/halfgemm_sve.h"

static bool SveAvailable() {
  return MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve();
}

class MlasSveHGemmTransposeATest : public MlasTestBase {
 private:
  MatrixGuardBuffer<uint16_t> BufferA;
  MatrixGuardBuffer<uint16_t> BufferD;
  MatrixGuardBuffer<uint16_t> BufferDRef;
  std::mt19937 gen_{20240702};

  void TestOne(size_t CountY, size_t CountX, size_t lda) {
    ASSERT_GE(lda, CountY);

    auto fill = [this](uint16_t* p, size_t n) {
      std::uniform_int_distribution<uint32_t> d(0, 0xFFFF);
      for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint16_t>(d(gen_));
    };

    const uint16_t* A = BufferA.GetFilledBuffer(CountX * lda, fill);
    uint16_t* D = BufferD.GetBuffer(CountY * CountX, /*ZeroFill*/ true);
    uint16_t* Dref = BufferDRef.GetBuffer(CountY * CountX, /*ZeroFill*/ true);

    for (size_t m = 0; m < CountY; ++m) {
      for (size_t k = 0; k < CountX; ++k) {
        Dref[m * CountX + k] = A[k * lda + m];
      }
    }

    MlasHgemmTransposeA_sve(reinterpret_cast<_mlas_fp16_*>(D),
                            reinterpret_cast<const _mlas_fp16_*>(A),
                            lda, CountY, CountX);

    for (size_t m = 0; m < CountY; ++m) {
      for (size_t k = 0; k < CountX; ++k) {
        const size_t idx = m * CountX + k;
        ASSERT_EQ(D[idx], Dref[idx])
            << "TransposeA mismatch at m=" << m << " k=" << k
            << " (CountY=" << CountY << " CountX=" << CountX << " lda=" << lda << ")";
      }
    }
  }

 public:
  static const char* GetTestSuiteName() { return "SveHGemmTransposeA"; }

  void ExecuteShort(void) override {
    if (!SveAvailable()) {
      GTEST_SKIP() << "SVE not available on this CPU.";
    }

    const size_t Ks[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32,
                         33, 40, 47, 48, 63, 64, 65, 96, 127, 128, 129, 200};
    for (size_t CountY = 1; CountY <= 13; ++CountY) {
      for (size_t CountX : Ks) {
        for (size_t pad : {size_t(0), size_t(1), size_t(5)}) {
          TestOne(CountY, CountX, CountY + pad);
        }
      }
    }
    TestOne(12, 512, 12);
    TestOne(12, 1023, 20);
    TestOne(8, 777, 8);
  }
};

class MlasSveHGemmEdgeTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<MLAS_FP16> BufferB;
  MatrixGuardBuffer<MLAS_FP16> BufferC;

  void TestZeroK(float beta) {
    constexpr size_t M = 5, N = 19, ldc = N + 3;
    const MLAS_FP16 nan = MLAS_FP16::FromBits(0x7E00);
    MLAS_FP16* C = BufferC.GetFilledBuffer(M * ldc, [&](MLAS_FP16* p, size_t n) {
      for (size_t i = 0; i < n; ++i) p[i] = (beta == 0.0f) ? nan : MLAS_FP16(float(i % 7));
    });
    std::vector<MLAS_FP16> expected(C, C + M * ldc);
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        expected[i * ldc + j] = MLAS_FP16(beta == 0.0f ? 0.0f : expected[i * ldc + j].ToFloat() * beta);
      }
    }

    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, 0, nullptr, 1, nullptr, N, C, ldc,
             MLAS_FP16(1.0f).val, MLAS_FP16(beta).val, nullptr);

    for (size_t i = 0; i < M * ldc; ++i) {
      ASSERT_EQ(C[i].val, expected[i].val) << "K == 0 mismatch at " << i << " beta=" << beta;
    }
  }

 public:
  static const char* GetTestSuiteName() { return "SveHGemmEdge"; }

  void ExecuteShort(void) override {
    if (!SveAvailable()) {
      GTEST_SKIP() << "SVE not available on this CPU.";
    }

    TestZeroK(0.0f);
    TestZeroK(0.5f);

    // Packing an empty B must not write to PackedB.
    const MLAS_FP16* B = BufferB.GetBuffer(64, /*ZeroFill*/ true);
    for (CBLAS_TRANSPOSE TransB : {CblasNoTrans, CblasTrans}) {
      MlasHGemmPackB(TransB, 0, 8, B, 8, nullptr);
      MlasHGemmPackB(TransB, 8, 0, B, 8, nullptr);
    }
  }
};

class MlasSveHalfGemmTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<MLAS_FP16> BufferA;
  MatrixGuardBuffer<MLAS_FP16> BufferB;
  MatrixGuardBuffer<MLAS_FP16> BufferC;
  MatrixGuardBuffer<float> BufferCReference;
  std::mt19937 gen_{13572468};

  void FillFp16(MLAS_FP16* p, size_t n) {
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (size_t i = 0; i < n; ++i) p[i] = MLAS_FP16(d(gen_));
  }

 public:
  static const char* GetTestSuiteName() { return "SveHalfGemm"; }

  void Test(size_t M, size_t N, size_t K, bool transA, bool transB, float alpha, float beta) {
    if (!SveAvailable()) {
      GTEST_SKIP() << "SVE not available on this CPU.";
    }
    if (!MlasHGemmSupported(transA ? CblasTrans : CblasNoTrans,
                            transB ? CblasTrans : CblasNoTrans)) {
      GTEST_SKIP() << "HGEMM not supported for this transpose combination.";
    }

    const size_t lda = (transA ? M : K) + 3;
    const size_t ldb = (transB ? K : N) + 5;
    const size_t ldc = N + 7;

    const size_t aRows = transA ? K : M;
    const size_t bRows = transB ? N : K;

    auto fill = [this](MLAS_FP16* p, size_t n) { FillFp16(p, n); };
    const MLAS_FP16* A = BufferA.GetFilledBuffer(aRows * lda, fill);
    const MLAS_FP16* B = BufferB.GetFilledBuffer(bRows * ldb, fill);
    MLAS_FP16* C = BufferC.GetFilledBuffer(M * ldc, fill);

    float* Cref = BufferCReference.GetBuffer(M * ldc, /*ZeroFill*/ true);
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        Cref[i * ldc + j] = C[i * ldc + j].ToFloat();
      }
    }

    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float accu = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          const float av = A[transA ? (k * lda + i) : (i * lda + k)].ToFloat();
          const float bv = B[transB ? (j * ldb + k) : (k * ldb + j)].ToFloat();
          accu += av * bv;
        }
        float& c = Cref[i * ldc + j];
        c = accu * alpha + c * beta;
      }
    }

    MlasGemm(transA ? CblasTrans : CblasNoTrans,
             transB ? CblasTrans : CblasNoTrans,
             M, N, K,
             A, lda, B, ldb, C, ldc,
             MLFp16(alpha).val, MLFp16(beta).val,
             nullptr);

    // fp16 accumulation error grows with K.
    const float rtol = 0.03f;
    const float atol = 0.06f + 0.0015f * static_cast<float>(K);
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        const float got = C[i * ldc + j].ToFloat();
        const float ref = Cref[i * ldc + j];
        ASSERT_LE(std::abs(got - ref), std::abs(ref) * rtol + atol)
            << "GEMM mismatch @[" << i << "," << j << "] got=" << got << " ref=" << ref
            << " M=" << M << " N=" << N << " K=" << K
            << " transA=" << transA << " transB=" << transB
            << " alpha=" << alpha << " beta=" << beta;
      }
    }
  }
};

class SveHalfGemmShortExecuteTest : public MlasTestFixture<MlasSveHalfGemmTest> {
 public:
  explicit SveHalfGemmShortExecuteTest(size_t M, size_t N, size_t K, bool transA, bool transB,
                                       float alpha, float beta)
      : M_(M), N_(N), K_(K), transA_(transA), transB_(transB), alpha_(alpha), beta_(beta) {}

  void TestBody() override {
    MlasTestFixture<MlasSveHalfGemmTest>::mlas_tester->Test(M_, N_, K_, transA_, transB_, alpha_, beta_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool transA, bool transB,
                                    float alpha, float beta) {
    std::stringstream ss;
    ss << (transA ? "TA" : "NA") << (transB ? "TB" : "NB")
       << "/M" << M << "xN" << N << "xK" << K
       << "/alpha" << alpha << "_beta" << beta;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasSveHalfGemmTest::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<MlasSveHalfGemmTest>* {
          return new SveHalfGemmShortExecuteTest(M, N, K, transA, transB, alpha, beta);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t count = 0;
    const bool trans[2] = {false, true};

    for (bool tA : trans) {
      for (bool tB : trans) {
        for (size_t M = 1; M <= 13; ++M) {
          count += RegisterSingleTest(M, 32, 33, tA, tB, 1.0f, 0.0f);
          count += RegisterSingleTest(M, 17, 65, tA, tB, 1.0f, 0.0f);
        }
        for (size_t K : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
          count += RegisterSingleTest(4, 40, K, tA, tB, 1.0f, 0.0f);
          count += RegisterSingleTest(12, 24, K, tA, tB, 1.0f, 0.0f);
        }
        count += RegisterSingleTest(7, 33, 40, tA, tB, 0.5f, 1.0f);
        count += RegisterSingleTest(9, 48, 64, tA, tB, 1.5f, 0.5f);
        count += RegisterSingleTest(13, 31, 63, tA, tB, 0.5f, 0.5f);
        count += RegisterSingleTest(2, 129, 128, tA, tB, 1.0f, 0.0f);
        count += RegisterSingleTest(33, 65, 96, tA, tB, 1.0f, 0.0f);
        count += RegisterSingleTest(64, 64, 512, tA, tB, 1.0f, 0.0f);
        count += RegisterSingleTest(129, 130, 257, tA, tB, 1.0f, 0.0f);
      }
    }
    return count;
  }

 private:
  size_t M_, N_, K_;
  bool transA_, transB_;
  float alpha_, beta_;
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (!SveAvailable()) {
    return false;
  }
  if (!is_short_execute) {
    return false;
  }
  size_t count = 0;
  count += MlasDirectShortExecuteTests<MlasSveHGemmTransposeATest>::RegisterShortExecute();
  count += MlasDirectShortExecuteTests<MlasSveHGemmEdgeTest>::RegisterShortExecute();
  count += SveHalfGemmShortExecuteTest::RegisterShortExecuteTests();
  return count > 0;
});

#endif  // defined(MLAS_USE_SVE) && defined(MLAS_TARGET_ARM64)
