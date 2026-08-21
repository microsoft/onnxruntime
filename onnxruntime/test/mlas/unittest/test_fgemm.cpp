// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

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
