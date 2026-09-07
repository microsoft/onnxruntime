/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    test_halfgemm_sve.cpp

Abstract:

    Correctness tests for the AArch64 SVE fp16 GEMM path (onnxruntime/core/mlas/
    lib/sve/halfgemm_kernel_sve.cpp, driven by hgemm.cpp).

    Two complementary suites are registered:

      1. SveHGemmTransposeA - a direct, bit-exact unit test of the
         MlasHgemmTransposeA_sve primitive (the TransA packing transpose). The
         transpose is pure data movement, so it is checked for an EXACT match
         against a scalar reference over an exhaustive sweep of the M (CountY)
         and K (CountX) dimensions plus leading-dimension padding. Inputs live
         in guard-page-backed buffers sized exactly to the operand, so any
         lane over-read or output over-write faults immediately. This is the
         kernel that was extended from the 256-bit-only fast path to also cover
         128-bit (svcnth()==8, with M-splitting) and 512-bit (svcnth()==32).

      2. SveHalfGemm - an end-to-end correctness sweep through the public
         MlasGemm fp16 API, exercising the whole SVE pipeline (TransposeA /
         CopyPackB / TransposePackB / KernelZero / KernelAdd) across all four
         (TransA, TransB) combinations, alpha/beta variants, a wide M/N/K sweep
         and padded strides. Registered one gtest per shape (SGEMM-style) so it
         is filterable via --gtest_filter and reads well in the runner output.

    Both suites are gated on runtime SVE availability and no-op otherwise.

--*/

#include "test_util.h"
#include "test_fp16.h"

#include "core/mlas/lib/mlasi.h"

#if defined(MLAS_USE_SVE) && defined(MLAS_TARGET_ARM64)

#include <sstream>
#include <vector>

//
// The SVE TransA transpose primitive. Defined (external linkage) in
// halfgemm_kernel_sve.cpp and declared locally by hgemm.cpp; re-declared here
// so the unit test can drive it directly. Layout contract:
//   D[m*CountX + k] = A[k*lda + m],  m in [0,CountY), k in [0,CountX)
// i.e. A is stored K-major with row stride lda, D is the row-major CountY x
// CountX NoTrans panel the compute kernels consume.
//
// The kernels are extern "C" so the intrinsics reference and the frozen
// machine code in aarch64/halfgemm_sve_asm.S are interchangeable; take the
// declaration from the header both implementations agree on rather than
// restating it here.
#include "core/mlas/lib/sve/halfgemm_sve.h"

static bool SveAvailable() {
  return MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve();
}

// ===========================================================================
//  Suite 1: direct bit-exact test of MlasHgemmTransposeA_sve
// ===========================================================================
class MlasSveHGemmTransposeATest : public MlasTestBase {
 private:
  MatrixGuardBuffer<uint16_t> BufferA;
  MatrixGuardBuffer<uint16_t> BufferD;
  MatrixGuardBuffer<uint16_t> BufferDRef;
  std::mt19937 gen_{20240702};

  // One case: transpose a CountY(M) x CountX(K) block whose A operand is stored
  // K-major with the given leading dimension (lda >= CountY). Buffers are sized
  // exactly to the operand so the guard page catches any over-read/over-write.
  void TestOne(size_t CountY, size_t CountX, size_t lda) {
    ASSERT_GE(lda, CountY);

    auto fill = [this](uint16_t* p, size_t n) {
      // Distinct-ish random fp16 bit patterns; correctness is by exact match
      // against the reference computed from the same buffer, so the actual
      // values only need to be varied enough that a mis-picked source element
      // is observable. Padding columns [CountY, lda) are filled too and must
      // never surface in the output.
      std::uniform_int_distribution<uint32_t> d(0, 0xFFFF);
      for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint16_t>(d(gen_));
    };

    const uint16_t* A = BufferA.GetFilledBuffer(CountX * lda, fill);
    uint16_t* D = BufferD.GetBuffer(CountY * CountX, /*ZeroFill*/ true);
    uint16_t* Dref = BufferDRef.GetBuffer(CountY * CountX, /*ZeroFill*/ true);

    // Scalar reference transpose.
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

    // CountY (M) is <= MLAS_HGEMM_TRANSA_ROWS (== 12) in production; sweep a
    // little past it. CountX (K) sweeps across and past each vector-length tile
    // boundary (8 / 16 / 32) so the vectorised tiles and the scalar K-remainder
    // are both exercised at every supported SVE width. lda padding forces the
    // predicated M-lane loads to be tested against a tight A row.
    const size_t Ks[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32,
                         33, 40, 47, 48, 63, 64, 65, 96, 127, 128, 129, 200};
    for (size_t CountY = 1; CountY <= 13; ++CountY) {
      for (size_t CountX : Ks) {
        for (size_t pad : {size_t(0), size_t(1), size_t(5)}) {
          TestOne(CountY, CountX, CountY + pad);
        }
      }
    }
    // A couple of larger panels.
    TestOne(12, 512, 12);
    TestOne(12, 1023, 20);
    TestOne(8, 777, 8);
  }
};

// ===========================================================================
//  Suite 2: end-to-end MlasGemm fp16 correctness (all Trans/alpha/beta)
// ===========================================================================
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

    // Padded leading dimensions (mirrors the NeonHGemm test conventions).
    const size_t lda = (transA ? M : K) + 3;
    const size_t ldb = (transB ? K : N) + 5;
    const size_t ldc = N + 7;

    const size_t aRows = transA ? K : M;
    const size_t bRows = transB ? N : K;

    auto fill = [this](MLAS_FP16* p, size_t n) { FillFp16(p, n); };
    const MLAS_FP16* A = BufferA.GetFilledBuffer(aRows * lda, fill);
    const MLAS_FP16* B = BufferB.GetFilledBuffer(bRows * ldb, fill);
    MLAS_FP16* C = BufferC.GetFilledBuffer(M * ldc, fill);

    // Capture the initial C (for the beta term) and build the float reference.
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

    // fp16 accumulation in the kernel diverges from the float reference as K
    // grows, so scale the absolute tolerance with K. A transpose/pack/kernel
    // indexing bug produces order-of-result errors, comfortably above this.
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

//
// Per-shape dynamic registration (SGEMM-style): one gtest per shape so the set
// is filterable and self-describing in the runner output.
//
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
        // Dense small-M sweep: M spans the TransA CountY range (1..12) and just
        // past it; N and K straddle vector-length tile boundaries.
        for (size_t M = 1; M <= 13; ++M) {
          count += RegisterSingleTest(M, 32, 33, tA, tB, 1.0f, 0.0f);
          count += RegisterSingleTest(M, 17, 65, tA, tB, 1.0f, 0.0f);
        }
        // K remainder / boundary sweep.
        for (size_t K : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
          count += RegisterSingleTest(4, 40, K, tA, tB, 1.0f, 0.0f);
          count += RegisterSingleTest(12, 24, K, tA, tB, 1.0f, 0.0f);
        }
        // alpha/beta variants.
        count += RegisterSingleTest(7, 33, 40, tA, tB, 0.5f, 1.0f);
        count += RegisterSingleTest(9, 48, 64, tA, tB, 1.5f, 0.5f);
        count += RegisterSingleTest(13, 31, 63, tA, tB, 0.5f, 0.5f);
        // A few larger / irregular shapes.
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
  count += SveHalfGemmShortExecuteTest::RegisterShortExecuteTests();
  return count > 0;
});

#endif  // defined(MLAS_USE_SVE) && defined(MLAS_TARGET_ARM64)
