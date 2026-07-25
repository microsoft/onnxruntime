/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_halfgemm.cpp

Abstract:

    Tests for MLAS half precision GEMM.

--*/

#include "test_halfgemm.h"
#include "core/mlas/lib/halfgemm.h"
#if defined(USE_KLEIDIAI)
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"
#endif

#include <array>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>

namespace {

struct HalfGemmPackBPaddingKernel {
  static constexpr size_t PackedK = 4;
};

void ExpectBufferFilledWith(const std::vector<std::byte>& buffer, std::byte expected) {
  const auto expected_value = std::to_integer<unsigned int>(expected);
  for (size_t i = 0; i < buffer.size(); ++i) {
    EXPECT_EQ(std::to_integer<unsigned int>(buffer[i]), expected_value) << "index=" << i;
  }
}

}  // namespace

#if defined(USE_KLEIDIAI)
namespace {

bool g_test_halfgemm_override_called = false;

bool MLASCALL
TestHalfGemmBatchOverride(
    size_t M,
    size_t N,
    size_t,
    size_t BatchN,
    const MLAS_HALF_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL*,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG*) {
  g_test_halfgemm_override_called = true;

  for (size_t batch = 0; batch < BatchN; ++batch) {
    auto* c = DataParams[batch].C;
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        c[m * DataParams[batch].ldc + n] = onnxruntime::MLFloat16(1.0f);
      }
    }
  }

  return true;
}

void ReferenceHalfGemm(
    size_t M,
    size_t N,
    size_t K,
    const MLFp16* A,
    const MLFp16* B,
    MLFp16* C) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += float(A[m * K + k]) * float(B[k * N + n]);
      }
      C[m * N + n] = MLFp16(sum);
    }
  }
}

struct HalfGemmOverrideGuard {
  explicit HalfGemmOverrideGuard(MLAS_HALF_GEMM_BATCH_OVERRIDE* replacement)
      : original_(GetMlasPlatform().MlasHalfGemmBatchOverride) {
    GetMlasPlatform().MlasHalfGemmBatchOverride = replacement;
  }

  ~HalfGemmOverrideGuard() {
    GetMlasPlatform().MlasHalfGemmBatchOverride = original_;
  }

 private:
  MLAS_HALF_GEMM_BATCH_OVERRIDE* original_;
};

}  // namespace

TEST(HalfGemmKleidiAISelector, DisableKleidiAIBypassesOverride) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP() << "HalfGemm FP16 acceleration not available.";
  }

  constexpr size_t M = 5;
  constexpr size_t N = 7;
  constexpr size_t K = 9;

  std::vector<MLFp16> A(M * K);
  std::vector<MLFp16> B(K * N);
  std::vector<MLFp16> C(M * N, MLFp16(0.0f));
  std::vector<MLFp16> CReference(M * N, MLFp16(0.0f));

  SmallFloatFill(A.data(), A.size());
  SmallFloatFill(B.data(), B.size());

  MLAS_HALF_GEMM_DATA_PARAMS data{};
  data.A = A.data();
  data.B = B.data();
  data.C = reinterpret_cast<MLAS_FP16*>(C.data());
  data.lda = K;
  data.ldb = N;
  data.ldc = N;

  HalfGemmOverrideGuard guard(TestHalfGemmBatchOverride);

  g_test_halfgemm_override_called = false;
  MlasHalfGemmBatch(M, N, K, 1, &data, nullptr);
  ASSERT_TRUE(g_test_halfgemm_override_called);
  for (const auto& value : C) {
    ASSERT_EQ(float(value), 1.0f);
  }

  std::fill(C.begin(), C.end(), MLFp16(0.0f));
  ReferenceHalfGemm(M, N, K, A.data(), B.data(), CReference.data());

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config;
  selector_config.use_kleidiai = false;
  data.BackendKernelSelectorConfig = &selector_config;

  g_test_halfgemm_override_called = false;
  MlasHalfGemmBatch(M, N, K, 1, &data, nullptr);
  ASSERT_FALSE(g_test_halfgemm_override_called);

  for (size_t i = 0; i < C.size(); ++i) {
    ASSERT_TRUE(CloseEnough(float(C[i]), float(CReference[i]))) << "index=" << i;
  }
}

TEST(HalfGemmKleidiAISelector, DisableKleidiAIInAnyBatchBypassesOverride) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP() << "HalfGemm FP16 acceleration not available.";
  }

  constexpr size_t M = 5;
  constexpr size_t N = 7;
  constexpr size_t K = 9;
  constexpr size_t BatchN = 2;

  std::vector<MLFp16> A(BatchN * M * K);
  std::vector<MLFp16> B(BatchN * K * N);
  std::vector<MLFp16> C(BatchN * M * N, MLFp16(0.0f));
  std::vector<MLFp16> CReference(BatchN * M * N, MLFp16(0.0f));
  SmallFloatFill(A.data(), A.size());
  SmallFloatFill(B.data(), B.size());

  std::array<MLAS_HALF_GEMM_DATA_PARAMS, BatchN> data{};
  for (size_t batch = 0; batch < BatchN; ++batch) {
    data[batch].A = A.data() + batch * M * K;
    data[batch].B = B.data() + batch * K * N;
    data[batch].C = reinterpret_cast<MLAS_FP16*>(C.data() + batch * M * N);
    data[batch].lda = K;
    data[batch].ldb = N;
    data[batch].ldc = N;
    ReferenceHalfGemm(M, N, K,
                      A.data() + batch * M * K,
                      B.data() + batch * K * N,
                      CReference.data() + batch * M * N);
  }

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config;
  selector_config.use_kleidiai = false;
  data[1].BackendKernelSelectorConfig = &selector_config;

  HalfGemmOverrideGuard guard(TestHalfGemmBatchOverride);
  g_test_halfgemm_override_called = false;
  MlasHalfGemmBatch(M, N, K, BatchN, data.data(), nullptr);
  ASSERT_FALSE(g_test_halfgemm_override_called);

  for (size_t i = 0; i < C.size(); ++i) {
    ASSERT_TRUE(CloseEnough(float(C[i]), float(CReference[i]))) << "index=" << i;
  }
}
#endif

namespace {

struct HalfGemmCase {
  const char* name;
  size_t M;
  size_t N;
  size_t K;
  size_t Batch;
  bool has_bias;
};

template <typename Runner>
void RunHalfGemmCases(const HalfGemmCase* test_cases, size_t num_cases, Runner run_case) {
  for (size_t i = 0; i < num_cases; ++i) {
    const auto& test_case = test_cases[i];
    SCOPED_TRACE(testing::Message()
                 << test_case.name
                 << " Batch=" << test_case.Batch
                 << " M=" << test_case.M
                 << " N=" << test_case.N
                 << " K=" << test_case.K
                 << " hasBias=" << test_case.has_bias);
    run_case(test_case);
  }
}

void ReferenceHalfGemmPackedCompatibility(
    size_t M,
    size_t N,
    size_t K,
    const MLFp16* A,
    const MLFp16* B,
    MLFp16* C) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        MLFp16 down(float(A[m * K + k]) * float(B[k * N + n]) + sum);
        sum = float(down);
      }
      C[m * N + n] = MLFp16(sum);
    }
  }
}

constexpr HalfGemmCase kNativeFp16Cases[] = {
    {"WideNBatch1WithBias", 43, 500, 401, 1, true},
    {"WideNBatch3NoBias", 43, 500, 401, 3, false},
    {"VectorLikeBatch3WithBias", 1, 32, 79, 3, true},
    {"RectangularBatch1WithBias", 64, 48, 80, 1, true},
    {"RectangularBatch1NoBias", 64, 48, 80, 1, false},
};

template <bool Threaded>
void RunNativeFp16WithoutOutputProcessorCases(const HalfGemmCase* test_cases, size_t num_cases) {
  RunHalfGemmCases(test_cases, num_cases, [](const HalfGemmCase& test_case) {
    MlasHalfGemmTest<MLFp16, MLFp16, false, Threaded> test;
    test.TestNativeFp16WithoutOutputProcessor(
        test_case.M, test_case.N, test_case.K, test_case.Batch, test_case.has_bias);
  });
}

constexpr HalfGemmCase kKleidiAIPathNonPackedCases[] = {
    {"WideNBatch3WithBias", 43, 500, 401, 3, true},
    {"WideNBatch3NoBias", 43, 500, 401, 3, false},
    {"RectangularBatch1WithBias", 64, 48, 80, 1, true},
    {"RectangularBatch1NoBias", 64, 48, 80, 1, false},
};

constexpr HalfGemmCase kKleidiAIPathPackedCases[] = {
    {"WideNBatch1WithBias", 43, 500, 401, 1, true},
    {"WideNBatch1NoBias", 43, 500, 401, 1, false},
    {"RectangularBatch1WithBias", 64, 48, 80, 1, true},
    {"RectangularBatch1NoBias", 64, 48, 80, 1, false},
};

template <typename AType, typename BType, bool Packed>
void RunKleidiAIWithoutOutputProcessorCases(const HalfGemmCase* test_cases, size_t num_cases) {
  RunHalfGemmCases(test_cases, num_cases, [](const HalfGemmCase& test_case) {
    MlasHalfGemmTest<AType, BType, Packed, false> test;
    test.TestKleidiAIWithoutOutputProcessor(
        test_case.M, test_case.N, test_case.K, test_case.Batch, test_case.has_bias);
  });
}

}  // namespace

TEST(HalfGemm, ZeroKInitializesBiasAndRunsOutputProcessor) {
  constexpr size_t M = 3;
  constexpr size_t N = 4;
  constexpr size_t K = 0;

  std::vector<MLFp16> Bias{MLFp16(1.0f), MLFp16(-2.0f), MLFp16(3.5f), MLFp16(0.25f)};
  std::vector<MLFp16> C(M * N, MLFp16(-9.0f));
  std::vector<float> CFloat(M * N, -9.0f);

  MLAS_ACTIVATION act;
  act.ActivationKind = MlasIdentityActivation;
  MLAS_HALF_GEMM_2FLOAT_PROCESSOR output_processor(act, CFloat.data(), N);

  MLAS_HALF_GEMM_DATA_PARAMS data{};
  data.Bias = reinterpret_cast<const MLAS_FP16*>(Bias.data());
  data.C = reinterpret_cast<MLAS_FP16*>(C.data());
  data.ldc = N;
  data.AIsfp32 = true;
  data.OutputProcessor = &output_processor;

  MlasHalfGemmBatch(M, N, K, 1, &data, nullptr);

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      const size_t index = m * N + n;
      ASSERT_EQ(float(C[index]), float(Bias[n])) << "index=" << index;
      ASSERT_EQ(CFloat[index], float(Bias[n])) << "index=" << index;
    }
  }
}

TEST(HalfGemm, ZeroKInitializesZeroWithoutBias) {
  constexpr size_t M = 3;
  constexpr size_t N = 4;
  constexpr size_t K = 0;

  std::vector<MLFp16> C(M * N, MLFp16(-9.0f));

  MLAS_HALF_GEMM_DATA_PARAMS data{};
  data.C = reinterpret_cast<MLAS_FP16*>(C.data());
  data.ldc = N;
  data.BIsfp32 = true;

  MlasHalfGemmBatch(M, N, K, 1, &data, nullptr);

  for (const auto& value : C) {
    ASSERT_EQ(float(value), 0.0f);
  }
}

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <typename AType, typename BType, bool Packed, bool Threaded>
class HalfGemmShortExecuteTest : public MlasTestFixture<MlasHalfGemmTest<AType, BType, Packed, Threaded>> {
 public:
  explicit HalfGemmShortExecuteTest(size_t M, size_t N, size_t K, size_t Batch, bool hasBias)
      : M_(M), N_(N), K_(K), Batch_(Batch), hasBias_(hasBias) {}

  void TestBody() override {
    MlasTestFixture<MlasHalfGemmTest<AType, BType, Packed, Threaded>>::mlas_tester->Test(M_, N_, K_, Batch_, hasBias_);
  }

  static size_t RegisterSingleTest(size_t M, size_t N, size_t K, size_t Batch, bool hasBias) {
    std::stringstream ss;
    ss << "Batch" << Batch << "/M" << M << "xN" << N << "xK" << K << "/"
       << "hasBias" << hasBias;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasHalfGemmTest<AType, BType, Packed, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasHalfGemmTest<AType, BType, Packed, Threaded>>* {
          return new HalfGemmShortExecuteTest<AType, BType, Packed, Threaded>(
              M, N, K, Batch, hasBias);
        });

    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (size_t b = 1; b < 16; b++) {
      test_registered += RegisterSingleTest(b, b, b, 1, false);
      test_registered += RegisterSingleTest(b, b, b, 1, true);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += RegisterSingleTest(b, b, b, 1, false);
      test_registered += RegisterSingleTest(b, b, b, 1, true);
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += RegisterSingleTest(b, b, b, 1, true);
    }
    for (size_t b = 1; b < 96; b++) {
      test_registered += RegisterSingleTest(1, b, 32, 1, false);
      test_registered += RegisterSingleTest(1, 32, b, 1, true);
      test_registered += RegisterSingleTest(1, b, b, 1, false);
      if (!Packed) {
        test_registered += RegisterSingleTest(1, b, 32, 3, true);
        test_registered += RegisterSingleTest(1, 32, b, 5, false);
      }
    }
    test_registered += RegisterSingleTest(43, 500, 401, 1, true);
    //    test_registered += RegisterSingleTest(1001, 1027, 1031, 1, false);
    if (!Packed) {
      test_registered += RegisterSingleTest(43, 500, 401, 5, true);
      //      test_registered += RegisterSingleTest(1000, 1029, 1030, 3, false);
    }

    return test_registered;
  }

 private:
  size_t M_, N_, K_, Batch_;
  bool hasBias_;
};

static size_t HalfGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, false, false>>::RegisterLongExecute();
  if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, true, false>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, true, false>>::RegisterLongExecute();
  }
  if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, true, false>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, true, false>>::RegisterLongExecute();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, false, true>>::RegisterLongExecute();
    if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
      count += MlasLongExecuteTests<MlasHalfGemmTest<float, MLFp16, true, true>>::RegisterLongExecute();
      count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, MLFp16, true, true>>::RegisterLongExecute();
    }
    if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
      count += MlasLongExecuteTests<MlasHalfGemmTest<MLFp16, float, true, true>>::RegisterLongExecute();
      count += MlasLongExecuteTests<MlasHalfGemmTest<float, float, true, true>>::RegisterLongExecute();
    }
  }

  return count;
}

static size_t HalfGemmRegistShortExecute() {
  size_t count = 0;

  count += HalfGemmShortExecuteTest<float, float, false, false>::RegisterShortExecuteTests();
  count += HalfGemmShortExecuteTest<MLFp16, float, false, false>::RegisterShortExecuteTests();
  count += HalfGemmShortExecuteTest<float, MLFp16, false, false>::RegisterShortExecuteTests();
  count += HalfGemmShortExecuteTest<MLFp16, MLFp16, false, false>::RegisterShortExecuteTests();
  if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
    count += HalfGemmShortExecuteTest<MLFp16, MLFp16, true, false>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<float, MLFp16, true, false>::RegisterShortExecuteTests();
  }
  if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
    count += HalfGemmShortExecuteTest<MLFp16, float, true, false>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<float, float, true, false>::RegisterShortExecuteTests();
  }

  if (GetMlasThreadPool() != nullptr) {
    count += HalfGemmShortExecuteTest<float, float, false, true>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<MLFp16, float, false, true>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<float, MLFp16, false, true>::RegisterShortExecuteTests();
    count += HalfGemmShortExecuteTest<MLFp16, MLFp16, false, true>::RegisterShortExecuteTests();
    if (MlasHalfGemmPackBSize(128, 128, false) > 0) {
      count += HalfGemmShortExecuteTest<MLFp16, MLFp16, true, true>::RegisterShortExecuteTests();
      count += HalfGemmShortExecuteTest<float, MLFp16, true, true>::RegisterShortExecuteTests();
    }
    if (MlasHalfGemmPackBSize(128, 128, true) > 0) {
      count += HalfGemmShortExecuteTest<MLFp16, float, true, true>::RegisterShortExecuteTests();
      count += HalfGemmShortExecuteTest<float, float, true, true>::RegisterShortExecuteTests();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  if (!MlasFp16AccelerationSupported()) {
    return false;
  }
  if (is_short_execute) {
    return HalfGemmRegistShortExecute() > 0;
  }
  return HalfGemmRegistLongExecute() > 0;
});

TEST(HalfGemmPackB, ReturnsZeroOnOverflow) {
  const size_t max = (std::numeric_limits<size_t>::max)();
  EXPECT_EQ(MlasHalfGemmPackBSize(1, max, true), size_t{0});
  EXPECT_EQ(MlasHalfGemmPackBSize(max, 2, true), size_t{0});
}

TEST(HalfGemmPackB, PackBLeavesOutputUnchangedOnOverflow) {
  const size_t max = (std::numeric_limits<size_t>::max)();
  std::vector<MLAS_FP16> b(1);
  constexpr std::byte sentinel{0x5A};
  std::vector<std::byte> packed_b(16, sentinel);

  MlasHalfGemmPackB(max, 2, b.data(), max, packed_b.data());
  ExpectBufferFilledWith(packed_b, sentinel);
}

TEST(HalfGemmPackB, PackBExitsEarlyForInvalidArguments) {
  constexpr size_t N = 5;
  constexpr size_t K = 3;
  constexpr std::byte sentinel{0x5A};

  std::vector<MLAS_FP16> b(K * N);
  std::vector<std::byte> packed_b(MlasHalfGemmPackBSize(N, K, false), sentinel);
  ASSERT_FALSE(packed_b.empty());

  MlasHalfGemmPackB(N, K, b.data(), N - 1, packed_b.data());
  ExpectBufferFilledWith(packed_b, sentinel);

  MlasHalfGemmPackB(N, K, nullptr, N, packed_b.data());
  ExpectBufferFilledWith(packed_b, sentinel);

  MlasHalfGemmPackB(N, K, b.data(), N, nullptr);
  ExpectBufferFilledWith(packed_b, sentinel);
}

TEST(HalfGemmPackB, CopyPackBZeroPadsAlignedKTail) {
  constexpr size_t N = 5;
  constexpr size_t K = 3;
  constexpr size_t AlignedK = 4;

  std::vector<_mlas_fp16_> b(K * N);
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<_mlas_fp16_>(i + 1);
  }

  constexpr _mlas_fp16_ stale_tail_value = 0xFFFF;
  std::vector<_mlas_fp16_> packed(AlignedK * N, stale_tail_value);

  MlasHalfGemmCopyPackB<HalfGemmPackBPaddingKernel>(
      packed.data(),
      b.data(),
      N,
      N,
      K);

  for (size_t n = 0; n < N; ++n) {
    EXPECT_EQ(packed[K * N + n], 0) << "n=" << n;
  }
}

TEST(HalfGemmPackB, GenericPackedBFlagRunsOnFallback) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  constexpr size_t M = 5;
  constexpr size_t N = 7;
  constexpr size_t K = 9;

  std::vector<MLFp16> A(M * K);
  std::vector<MLFp16> B(K * N);
  std::vector<MLFp16> C(M * N, MLFp16(0.0f));
  std::vector<MLFp16> CReference(M * N, MLFp16(0.0f));

  SmallFloatFill(A.data(), A.size());
  SmallFloatFill(B.data(), B.size());

  const size_t packed_b_size = MlasHalfGemmPackBSize(N, K, false);
  if (packed_b_size == 0) {
    GTEST_SKIP();
  }

  std::vector<std::byte> packed_b(packed_b_size);
  MlasHalfGemmPackB(N, K, reinterpret_cast<const MLAS_FP16*>(B.data()), N, packed_b.data());

  MLAS_HALF_GEMM_DATA_PARAMS data{};
  data.A = A.data();
  data.B = packed_b.data();
  data.C = reinterpret_cast<MLAS_FP16*>(C.data());
  data.lda = K;
  data.ldb = 0;
  data.ldc = N;
  data.BIsPacked = true;

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};
  selector_config.use_kleidiai = false;
  data.BackendKernelSelectorConfig = &selector_config;

  MlasHalfGemmBatch(M, N, K, 1, &data, nullptr);
  ReferenceHalfGemmPackedCompatibility(M, N, K, A.data(), B.data(), CReference.data());

  for (size_t i = 0; i < C.size(); ++i) {
    ASSERT_TRUE(CloseEnough(float(C[i]), float(CReference[i]))) << "index=" << i;
  }
}

TEST(HalfGemmKleidiAINativeFp16, NoPackSingleThreadWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<MLFp16, MLFp16, false, false> test;
  test.TestNativeFp16WithoutOutputProcessor(43, 500, 401, 1, true);
}

TEST(HalfGemmKleidiAINativeFp16, NoPackSingleThreadWithoutOutputProcessorBatch3) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<MLFp16, MLFp16, false, false> test;
  test.TestNativeFp16WithoutOutputProcessor(43, 500, 401, 3, true);
}

TEST(HalfGemmKleidiAINativeFp16, NoPackThreadedWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (GetMlasThreadPool() == nullptr) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<MLFp16, MLFp16, false, true> test;
  test.TestNativeFp16WithoutOutputProcessor(43, 500, 401, 1, true);
}

TEST(HalfGemmKleidiAINativeFp16, NoPackThreadedWithoutOutputProcessorBatch3) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (GetMlasThreadPool() == nullptr) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<MLFp16, MLFp16, false, true> test;
  test.TestNativeFp16WithoutOutputProcessor(43, 500, 401, 3, true);
}

TEST(HalfGemmKleidiAINativeFp16, NoPackSingleThreadWithoutOutputProcessorVariedShapesAndBias) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  RunNativeFp16WithoutOutputProcessorCases<false>(kNativeFp16Cases, std::size(kNativeFp16Cases));
}

TEST(HalfGemmKleidiAINativeFp16, NoPackThreadedWithoutOutputProcessorVariedShapesAndBias) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (GetMlasThreadPool() == nullptr) {
    GTEST_SKIP();
  }

  RunNativeFp16WithoutOutputProcessorCases<true>(kNativeFp16Cases, std::size(kNativeFp16Cases));
}

TEST(HalfGemmKleidiAIPath, Fp32AConversionSingleThreadBatch3WithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<float, MLFp16, false, false> test;
  test.TestKleidiAIWithoutOutputProcessor(43, 500, 401, 3, true);
}

TEST(HalfGemmKleidiAIPath, Fp32AConversionSingleThreadVariedShapesAndBiasWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  RunKleidiAIWithoutOutputProcessorCases<float, MLFp16, false>(
      kKleidiAIPathNonPackedCases, std::size(kKleidiAIPathNonPackedCases));
}

TEST(HalfGemmKleidiAIPath, Fp32BConversionSingleThreadBatch3WithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<MLFp16, float, false, false> test;
  test.TestKleidiAIWithoutOutputProcessor(43, 500, 401, 3, true);
}

TEST(HalfGemmKleidiAIPath, Fp32BConversionSingleThreadVariedShapesAndBiasWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  RunKleidiAIWithoutOutputProcessorCases<MLFp16, float, false>(
      kKleidiAIPathNonPackedCases, std::size(kKleidiAIPathNonPackedCases));
}

TEST(HalfGemmKleidiAIPath, Fp32ABConversionSingleThreadBatch3WithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<float, float, false, false> test;
  test.TestKleidiAIWithoutOutputProcessor(43, 500, 401, 3, true);
}

TEST(HalfGemmKleidiAIPath, Fp32ABConversionSingleThreadVariedShapesAndBiasWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }

  RunKleidiAIWithoutOutputProcessorCases<float, float, false>(
      kKleidiAIPathNonPackedCases, std::size(kKleidiAIPathNonPackedCases));
}

TEST(HalfGemmKleidiAIPath, PackedBFp16SingleThreadWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (MlasHalfGemmPackBSize(128, 128, false) == 0) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<MLFp16, MLFp16, true, false> test;
  test.TestKleidiAIWithoutOutputProcessor(43, 500, 401, 1, true);
}

TEST(HalfGemmKleidiAIPath, PackedBFp16SingleThreadVariedShapesAndBiasWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (MlasHalfGemmPackBSize(128, 128, false) == 0) {
    GTEST_SKIP();
  }

  RunKleidiAIWithoutOutputProcessorCases<MLFp16, MLFp16, true>(
      kKleidiAIPathPackedCases, std::size(kKleidiAIPathPackedCases));
}

TEST(HalfGemmKleidiAIPath, PackedBFloatSingleThreadWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (MlasHalfGemmPackBSize(128, 128, true) == 0) {
    GTEST_SKIP();
  }

  MlasHalfGemmTest<MLFp16, float, true, false> test;
  test.TestKleidiAIWithoutOutputProcessor(43, 500, 401, 1, true);
}

TEST(HalfGemmKleidiAIPath, PackedBFloatSingleThreadVariedShapesAndBiasWithoutOutputProcessor) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (MlasHalfGemmPackBSize(128, 128, true) == 0) {
    GTEST_SKIP();
  }

  RunKleidiAIWithoutOutputProcessorCases<MLFp16, float, true>(
      kKleidiAIPathPackedCases, std::size(kKleidiAIPathPackedCases));
}

#if defined(USE_KLEIDIAI)
// KleidiAI-specific packed-B uses a separate direct-consumption contract from
// generic halfgemm PackB. Unsupported combinations fail at the public API
// boundary because generic MLAS cannot consume this backend-native layout.
TEST(HalfGemmKleidiAIPath, KleidiAIPackedBWithBiasIsRejected) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (GetMlasPlatform().MlasHalfGemmBatchOverride == nullptr) {
    GTEST_SKIP() << "KleidiAI halfgemm override unavailable";
  }

  constexpr size_t M = 5;
  constexpr size_t N = 7;
  constexpr size_t K = 9;

  std::vector<MLFp16> A(M * K);
  std::vector<MLFp16> B(K * N);
  std::vector<MLFp16> Bias(N);
  std::vector<MLFp16> C(M * N, MLFp16(0.0f));

  SmallFloatFill(A.data(), A.size());
  SmallFloatFill(B.data(), B.size());
  SmallFloatFill(Bias.data(), Bias.size());

  const size_t packed_b_size = ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(CblasNoTrans, CblasNoTrans, N, K);
  ASSERT_NE(packed_b_size, size_t{0});

  std::vector<std::byte> packed_b(packed_b_size);
  ASSERT_TRUE(ArmKleidiAI::MlasHalfGemmKleidiAIPackB(
      CblasNoTrans, CblasNoTrans, N, K, reinterpret_cast<const MLAS_FP16*>(B.data()), N, packed_b.data()));

  MLAS_HALF_GEMM_DATA_PARAMS data{};
  data.A = A.data();
  data.B = packed_b.data();
  data.Bias = reinterpret_cast<const MLAS_FP16*>(Bias.data());
  data.C = reinterpret_cast<MLAS_FP16*>(C.data());
  data.lda = K;
  data.ldb = 0;
  data.ldc = N;
  data.BIsBackendNativePacked = true;

  ASSERT_FALSE(ArmKleidiAI::MlasHalfGemmBatch(M, N, K, 1, &data, nullptr, nullptr));
#if !defined(ORT_NO_EXCEPTIONS)
  EXPECT_THROW(MlasHalfGemmBatch(M, N, K, 1, &data, nullptr), std::runtime_error);
#endif
}

TEST(HalfGemmKleidiAIPath, KleidiAIPackedBWithOutputProcessorIsRejected) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP();
  }
  if (GetMlasPlatform().MlasHalfGemmBatchOverride == nullptr) {
    GTEST_SKIP() << "KleidiAI halfgemm override unavailable";
  }

  constexpr size_t M = 5;
  constexpr size_t N = 7;
  constexpr size_t K = 9;

  std::vector<MLFp16> A(M * K);
  std::vector<MLFp16> B(K * N);
  std::vector<MLFp16> C(M * N, MLFp16(0.0f));
  std::vector<float> CFloat(M * N, 0.0f);

  SmallFloatFill(A.data(), A.size());
  SmallFloatFill(B.data(), B.size());

  const size_t packed_b_size = ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(CblasNoTrans, CblasNoTrans, N, K);
  ASSERT_NE(packed_b_size, size_t{0});

  std::vector<std::byte> packed_b(packed_b_size);
  ASSERT_TRUE(ArmKleidiAI::MlasHalfGemmKleidiAIPackB(
      CblasNoTrans, CblasNoTrans, N, K, reinterpret_cast<const MLAS_FP16*>(B.data()), N, packed_b.data()));

  MLAS_ACTIVATION act;
  act.ActivationKind = MlasIdentityActivation;
  MLAS_HALF_GEMM_2FLOAT_PROCESSOR output_processor(act, CFloat.data(), N);

  MLAS_HALF_GEMM_DATA_PARAMS data{};
  data.A = A.data();
  data.B = packed_b.data();
  data.C = reinterpret_cast<MLAS_FP16*>(C.data());
  data.lda = K;
  data.ldb = 0;
  data.ldc = N;
  data.BIsBackendNativePacked = true;
  data.OutputProcessor = &output_processor;

  ASSERT_FALSE(ArmKleidiAI::MlasHalfGemmBatch(M, N, K, 1, &data, nullptr, nullptr));
#if !defined(ORT_NO_EXCEPTIONS)
  EXPECT_THROW(MlasHalfGemmBatch(M, N, K, 1, &data, nullptr), std::runtime_error);
#endif
}

TEST(HalfGemmKleidiAIPath, ZeroKIsNotHandledByKleidiAIOverride) {
  if (GetMlasPlatform().MlasHalfGemmBatchOverride == nullptr) {
    GTEST_SKIP() << "KleidiAI halfgemm override unavailable";
  }

  constexpr size_t M = 5;
  constexpr size_t N = 7;
  constexpr size_t K = 0;

  std::vector<MLFp16> Bias(N);
  std::vector<MLFp16> C(M * N, MLFp16(1.0f));

  SmallFloatFill(Bias.data(), Bias.size());

  MLAS_HALF_GEMM_DATA_PARAMS data{};
  data.Bias = reinterpret_cast<const MLAS_FP16*>(Bias.data());
  data.C = reinterpret_cast<MLAS_FP16*>(C.data());
  data.ldc = N;

  const bool handled = ArmKleidiAI::MlasHalfGemmBatch(M, N, K, 1, &data, nullptr, nullptr);
  ASSERT_FALSE(handled);
}

TEST(HalfGemmKleidiAIPath, KleidiAIPackedBSizeRejectsUnsupportedTranspose) {
  if (GetMlasPlatform().MlasHalfGemmBatchOverride == nullptr) {
    GTEST_SKIP() << "KleidiAI halfgemm override unavailable";
  }

  constexpr size_t N = 7;
  constexpr size_t K = 9;

  EXPECT_EQ(ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(CblasTrans, CblasNoTrans, N, K), size_t{0});
  EXPECT_EQ(ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(CblasNoTrans, CblasTrans, N, K), size_t{0});
}

TEST(HalfGemmKleidiAIPath, KleidiAIPackedBRejectsUnsupportedTranspose) {
  if (GetMlasPlatform().MlasHalfGemmBatchOverride == nullptr) {
    GTEST_SKIP() << "KleidiAI halfgemm override unavailable";
  }

  constexpr size_t N = 7;
  constexpr size_t K = 9;

  std::vector<MLFp16> B(K * N);
  SmallFloatFill(B.data(), B.size());

  const size_t packed_b_size = ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(CblasNoTrans, CblasNoTrans, N, K);
  ASSERT_NE(packed_b_size, size_t{0});

  std::vector<std::byte> packed_b(packed_b_size);
  EXPECT_FALSE(ArmKleidiAI::MlasHalfGemmKleidiAIPackB(
      CblasTrans, CblasNoTrans, N, K, reinterpret_cast<const MLAS_FP16*>(B.data()), N, packed_b.data()));
  EXPECT_FALSE(ArmKleidiAI::MlasHalfGemmKleidiAIPackB(
      CblasNoTrans, CblasTrans, N, K, reinterpret_cast<const MLAS_FP16*>(B.data()), N, packed_b.data()));
}

TEST(HalfGemmKleidiAIPath, KleidiAIPackedBRejectsInvalidLeadingDimension) {
  if (GetMlasPlatform().MlasHalfGemmBatchOverride == nullptr) {
    GTEST_SKIP() << "KleidiAI halfgemm override unavailable";
  }

  constexpr size_t N = 7;
  constexpr size_t K = 9;

  std::vector<MLFp16> B(K * N);
  SmallFloatFill(B.data(), B.size());

  const size_t packed_b_size = ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(CblasNoTrans, CblasNoTrans, N, K);
  ASSERT_NE(packed_b_size, size_t{0});

  std::vector<std::byte> packed_b(packed_b_size);
  EXPECT_FALSE(ArmKleidiAI::MlasHalfGemmKleidiAIPackB(
      CblasNoTrans, CblasNoTrans, N, K, reinterpret_cast<const MLAS_FP16*>(B.data()), N - 1, packed_b.data()));
}
#endif
