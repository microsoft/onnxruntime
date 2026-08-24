// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"
#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI)
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"
#endif

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace {

void RunPublicSgemm(CBLAS_TRANSPOSE trans_b,
                    bool packed,
                    size_t m,
                    size_t n,
                    size_t k,
                    float alpha,
                    float beta,
                    const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& c) {
  std::vector<uint8_t> packed_b;
  const float* b_data = b.data();
  const size_t ldb = trans_b == CblasNoTrans ? n : k;

  if (packed) {
    const size_t packed_b_size = MlasGemmPackBSize(CblasNoTrans, trans_b, n, k, nullptr);
    ASSERT_GT(packed_b_size, 0u);
    packed_b.resize(packed_b_size);
    MlasGemmPackB(CblasNoTrans, trans_b, n, k, b.data(), ldb, packed_b.data(), nullptr);
    b_data = reinterpret_cast<const float*>(packed_b.data());
  }

  MLAS_SGEMM_DATA_PARAMS data;
  data.A = a.data();
  data.lda = k;
  data.B = b_data;
  data.ldb = ldb;
  data.C = c.data();
  data.ldc = n;
  data.alpha = alpha;
  data.beta = beta;
  data.BIsPacked = packed;

  MlasGemmBatch(CblasNoTrans, trans_b, m, n, k, &data, 1, nullptr, nullptr);
}

template <typename Verify>
void ForEachRhsLayoutAndPacking(Verify verify) {
  for (CBLAS_TRANSPOSE trans_b : {CblasNoTrans, CblasTrans}) {
    for (bool packed : {false, true}) {
      SCOPED_TRACE(testing::Message() << "TransB=" << static_cast<int>(trans_b) << " packed=" << packed);
      verify(trans_b, packed);
    }
  }
}

TEST(SGemmPublicApi, BetaZeroDoesNotReadC) {
  constexpr size_t M = 33;
  constexpr size_t N = 65;
  constexpr size_t K = 31;

  ForEachRhsLayoutAndPacking([](CBLAS_TRANSPOSE trans_b, bool packed) {
    std::vector<float> a(M * K, 2.0f);
    std::vector<float> b(K * N, 0.5f);
    std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());

    RunPublicSgemm(trans_b, packed, M, N, K, 0.5f, 0.0f, a, b, c);
    for (float value : c) {
      EXPECT_FLOAT_EQ(value, 15.5f);
    }
  });
}

TEST(SGemmPublicApi, AlphaZeroDoesNotReadInputs) {
  constexpr size_t M = 33;
  constexpr size_t N = 65;
  constexpr size_t K = 31;

  ForEachRhsLayoutAndPacking([](CBLAS_TRANSPOSE trans_b, bool packed) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> a(M * K, nan);
    std::vector<float> b(K * N, nan);
    std::vector<float> c(M * N, 4.0f);

    RunPublicSgemm(trans_b, packed, M, N, K, 0.0f, -0.5f, a, b, c);
    for (float value : c) {
      EXPECT_FLOAT_EQ(value, -2.0f);
    }
  });
}

TEST(SGemmPublicApi, InfiniteResultsAreNotClamped) {
  constexpr size_t M = 33;
  constexpr size_t N = 65;
  constexpr size_t K = 1;

  ForEachRhsLayoutAndPacking([](CBLAS_TRANSPOSE trans_b, bool packed) {
    std::vector<float> a(M * K);
    for (size_t row = 0; row < M; ++row) {
      a[row] = row % 2 == 0 ? std::numeric_limits<float>::infinity()
                            : -std::numeric_limits<float>::infinity();
    }
    std::vector<float> b(K * N, 1.0f);
    std::vector<float> c(M * N, 0.0f);

    RunPublicSgemm(trans_b, packed, M, N, K, 1.0f, 0.0f, a, b, c);
    for (size_t row = 0; row < M; ++row) {
      for (size_t col = 0; col < N; ++col) {
        const float value = c[row * N + col];
        EXPECT_TRUE(std::isinf(value));
        EXPECT_EQ(std::signbit(value), row % 2 != 0);
      }
    }
  });
}

TEST(SGemmPublicApi, MixedPackedAndUnpackedBatchSme2) {
#if defined(MLAS_TARGET_ARM64) && defined(USE_KLEIDIAI)
  if (!ArmKleidiAI::UseSME2) {
    GTEST_SKIP() << "Mixed KleidiAI SGEMM packing is exercised only by the SME2 path.";
  }
#else
  GTEST_SKIP() << "Mixed KleidiAI SGEMM packing requires an ARM64 SME2 build.";
#endif

  constexpr size_t M = 3;
  constexpr size_t N = 513;
  constexpr size_t K = 7;

  for (CBLAS_TRANSPOSE trans_b : {CblasNoTrans, CblasTrans}) {
    const size_t ldb = trans_b == CblasNoTrans ? N : K;
    for (size_t packed_index : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(testing::Message() << "TransB=" << static_cast<int>(trans_b)
                                      << " packed_index=" << packed_index);

      std::array<std::vector<float>, 2> a;
      std::array<std::vector<float>, 2> b;
      std::array<std::vector<float>, 2> c;
      std::array<std::vector<float>, 2> expected;
      for (size_t batch = 0; batch < a.size(); ++batch) {
        a[batch].resize(M * K);
        b[batch].resize(K * N);
        c[batch].assign(M * N, 0.0f);
        expected[batch].resize(M * N);

        for (size_t row = 0; row < M; ++row) {
          for (size_t k = 0; k < K; ++k) {
            const int value = static_cast<int>((row * 3 + k * 5 + batch * 7) % 17) - 8;
            a[batch][row * K + k] = static_cast<float>(value) / 8.0f;
          }
        }
        for (size_t k = 0; k < K; ++k) {
          for (size_t col = 0; col < N; ++col) {
            const int value = static_cast<int>((k * 7 + col * 3 + batch * 5) % 19) - 9;
            const size_t index = trans_b == CblasNoTrans ? k * N + col : col * K + k;
            b[batch][index] = static_cast<float>(value) / 16.0f;
          }
        }
        for (size_t row = 0; row < M; ++row) {
          for (size_t col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
              const size_t index = trans_b == CblasNoTrans ? k * N + col : col * K + k;
              sum += a[batch][row * K + k] * b[batch][index];
            }
            expected[batch][row * N + col] = sum;
          }
        }
      }

      const size_t packed_b_size = MlasGemmPackBSize(CblasNoTrans, trans_b, N, K, nullptr);
      ASSERT_GT(packed_b_size, 0u);
      std::vector<uint8_t> packed_b(packed_b_size);
      MlasGemmPackB(CblasNoTrans,
                    trans_b,
                    N,
                    K,
                    b[packed_index].data(),
                    ldb,
                    packed_b.data(),
                    nullptr);

      std::array<MLAS_SGEMM_DATA_PARAMS, 2> data{};
      for (size_t batch = 0; batch < data.size(); ++batch) {
        data[batch].A = a[batch].data();
        data[batch].lda = K;
        data[batch].B = batch == packed_index
                            ? reinterpret_cast<const float*>(packed_b.data())
                            : b[batch].data();
        data[batch].ldb = ldb;
        data[batch].C = c[batch].data();
        data[batch].ldc = N;
        data[batch].BIsPacked = batch == packed_index;
      }

      EXPECT_NO_THROW(
          MlasGemmBatch(
              CblasNoTrans, trans_b, M, N, K, data.data(), data.size(), GetMlasThreadPool(), nullptr));

      for (size_t batch = 0; batch < c.size(); ++batch) {
        for (size_t index = 0; index < c[batch].size(); ++index) {
          EXPECT_NEAR(c[batch][index], expected[batch][index], 1.0e-5f);
        }
      }
    }
  }
}

}  // namespace

static size_t FGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasFgemmTest<float, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasFgemmTest<float, true, false>>::RegisterLongExecute();

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasFgemmTest<float, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasFgemmTest<float, true, true>>::RegisterLongExecute();
  }

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE

  count += MlasLongExecuteTests<MlasFgemmTest<double, false, false>>::RegisterLongExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasFgemmTest<double, false, true>>::RegisterLongExecute();
  }

#endif

  return count;
}

static size_t FGemmRegistShortExecute() {
  size_t count = 0;

  count += FgemmShortExecuteTest<float, false, false>::RegisterShortExecuteTests();
  count += FgemmShortExecuteTest<float, true, false>::RegisterShortExecuteTests();

  if (GetMlasThreadPool() != nullptr) {
    count += FgemmShortExecuteTest<float, false, true>::RegisterShortExecuteTests();
    count += FgemmShortExecuteTest<float, true, true>::RegisterShortExecuteTests();
  }

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE

  count += FgemmShortExecuteTest<double, false, false>::RegisterShortExecuteTests();
  if (GetMlasThreadPool() != nullptr) {
    count += FgemmShortExecuteTest<double, false, true>::RegisterShortExecuteTests();
  }

#endif

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? FGemmRegistShortExecute() : FGemmRegistLongExecute();
});
