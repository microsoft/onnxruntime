// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// MlasSymmQgemmBatch selects Lit vs Big from IsCurrentCoreArmv8NarrowLd().
// Force each SDOT operation so any FEAT_DotProd host covers Ld64 (Lit) as
// well as the big-core kernel.

#include "test_symm_qgemm.h"

#if defined(MLAS_TARGET_ARM64)

#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/qgemm.h"

namespace {

constexpr char kSkipReason[] = "ARM64 FEAT_DotProd is not available, so MLAS does not use the SDOT SymmQgemm dispatch";

// Same check platform.cpp uses to install MlasSymmQgemmS8DispatchSdot. Gate on
// the CPU only, never on the dispatch contents, so a dispatch regression fails
// the tests below instead of skipping or unregistering them.
bool HasSdotSymmQgemmDispatch() { return MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot(); }

}  // namespace

TEST(SymmQgemmArm64Dispatch, DistinctSdotLitAndBigKernels) {
  if (!HasSdotSymmQgemmDispatch()) {
    GTEST_SKIP() << kSkipReason;
  }
  // The Neon fallback installs one kernel in both slots, so equal pointers on a
  // dotprod host mean the SDOT dispatch was not selected.
  const auto* dispatch = GetMlasPlatform().SymmQgemmDispatch;
  ASSERT_NE(dispatch, nullptr);
  ASSERT_NE(dispatch->LitOperation, nullptr);
  ASSERT_NE(dispatch->BigOperation, nullptr);
  ASSERT_NE(dispatch->LitOperation, dispatch->BigOperation);
}

class SymmQgemmArm64DispatchTest : public MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, false>> {
 public:
  enum class Kernel {
    kLit,
    kBig,
  };

  SymmQgemmArm64DispatchTest(size_t M, size_t N, size_t K, size_t Batch, int32_t offa, Kernel kernel,
                             bool signed_input)
      : M_(M), N_(N), K_(K), Batch_(Batch), offa_(offa), kernel_(kernel), signed_input_(signed_input) {}

  void TestBody() override {
    // Registered only on dotprod hosts, where a missing kernel is a regression.
    const auto* dispatch = GetMlasPlatform().SymmQgemmDispatch;
    ASSERT_NE(dispatch, nullptr);
    const MlasSymmQgemmForcedOperation operation =
        (kernel_ == Kernel::kLit) ? dispatch->LitOperation : dispatch->BigOperation;
    ASSERT_NE(operation, nullptr);
    auto* tester = MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, false>>::mlas_tester;

    if (!signed_input_) {
      tester->Test(M_, N_, K_, Batch_, offa_, operation);
      return;
    }

    // Same int8 extremes as SymmQgemmS8SignedInputTest.
    static const int8_t a_values[] = {-128, -1, 0, 1, 127, -64, 64, -33};
    static const int8_t b_values[] = {-128, -1, 0, 1, 127, -100, 100, 42};
    // Symmetric kernels may read up to 15 bytes past the logical end of A.
    constexpr size_t OVERRUN = 15;

    uint8_t* A = BufferA.GetFilledBuffer(M_ * K_ * Batch_ + OVERRUN, [](uint8_t* p, size_t n) {
      for (size_t i = 0; i < n; i++) {
        p[i] = uint8_t(a_values[i % _countof(a_values)]);
      }
    });
    int8_t* B = BufferB.GetFilledBuffer(K_ * N_ * Batch_, [](int8_t* p, size_t n) {
      for (size_t i = 0; i < n; i++) {
        p[i] = b_values[i % _countof(b_values)];
      }
    });
    int32_t* C = BufferC.GetBuffer(M_ * N_ * Batch_);
    int32_t* CReference = BufferCReference.GetBuffer(M_ * N_ * Batch_);

    tester->Test(M_, N_, K_, Batch_, A, K_, offa_, B, N_, C, CReference, N_, operation);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t Batch, int32_t offa, Kernel kernel,
                                   bool signed_input) {
    const char* suite = (kernel == Kernel::kLit) ? "SymmQgemmS8_Int32_Arm64Lit" : "SymmQgemmS8_Int32_Arm64Big";
    std::stringstream ss;
    ss << (signed_input ? "SignedInput/" : "DefaultFill/")
       << "Batch" << Batch << "/M" << M << "xN" << N << "xK" << K << "/offa" << offa;
    auto test_name = ss.str();

    testing::RegisterTest(
        suite,
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasSymmQgemmTest<int8_t, int32_t, false>>* {
          return new SymmQgemmArm64DispatchTest(M, N, K, Batch, offa, kernel, signed_input);
        });
    return 1;
  }

  static size_t RegisterKernelTests(Kernel kernel) {
    size_t test_registered = 0;

    static const size_t Ks[] = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33};
    static const size_t Ms[] = {1, 2, 3, 4, 5, 7, 8, 9};
    static const size_t Ns[] = {16, 32};
    static const int32_t offas[] = {0, -128, 127};

    for (size_t k = 0; k < _countof(Ks); k++) {
      for (size_t m = 0; m < _countof(Ms); m++) {
        for (size_t n = 0; n < _countof(Ns); n++) {
          for (size_t a = 0; a < _countof(offas); a++) {
            test_registered += RegisterSingleTest(Ms[m], Ns[n], Ks[k], 1, offas[a], kernel, true);
          }
        }
      }
    }

    // N tails around the 16-wide pack (not covered by the 16/32 signed grid).
    static const size_t TailNs[] = {1, 8, 17};
    static const size_t TailMs[] = {1, 4, 8};
    for (size_t k = 0; k < _countof(Ks); k++) {
      for (size_t m = 0; m < _countof(TailMs); m++) {
        for (size_t n = 0; n < _countof(TailNs); n++) {
          test_registered += RegisterSingleTest(TailMs[m], TailNs[n], Ks[k], 1, 0, kernel, true);
        }
      }
    }

    static const size_t DefaultShapes[][4] = {
        {1, 16, 16, 1},
        {4, 16, 16, 1},
        {8, 32, 32, 1},
        {43, 64, 65, 1},
        {5, 17, 33, 3},
    };
    for (size_t i = 0; i < _countof(DefaultShapes); i++) {
      test_registered += RegisterSingleTest(DefaultShapes[i][0], DefaultShapes[i][1], DefaultShapes[i][2],
                                            DefaultShapes[i][3], 21, kernel, false);
    }

    return test_registered;
  }

  static size_t RegisterShortExecuteTests() {
    if (!HasSdotSymmQgemmDispatch()) {
      return 0;
    }
    return RegisterKernelTests(Kernel::kLit) + RegisterKernelTests(Kernel::kBig);
  }

 private:
  MatrixGuardBuffer<uint8_t> BufferA;
  MatrixGuardBuffer<int8_t> BufferB;
  MatrixGuardBuffer<int32_t> BufferC;
  MatrixGuardBuffer<int32_t> BufferCReference;
  size_t M_, N_, K_, Batch_;
  int32_t offa_;
  Kernel kernel_;
  bool signed_input_;
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) -> size_t {
  return is_short_execute ? SymmQgemmArm64DispatchTest::RegisterShortExecuteTests() : 0;
});

#endif  // defined(MLAS_TARGET_ARM64)
