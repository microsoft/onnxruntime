// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

//
// Utility class for creating test vectors for vector dot prod tests.
//
template <typename T>
class TestVectors {
 public:
  TestVectors(const size_t M, const size_t N)
      : M(M), N(N) {
    for (size_t i = 0; i < M; ++i) {
      A.push_back(static_cast<T>(i + 1));
    }

    // Create B so that each row of A is the same with each row of B:
    B.resize(N * M);
    for (size_t i = 0; i < N; ++i) {
      size_t idx = i;
      for (size_t j = 0; j < M; ++j) {
        B[idx] = A[j];
        idx += N;
      }
    }

    C.resize(N);
  }

  TestVectors(const TestVectors& copy) = delete;
  TestVectors() = default;
  TestVectors& operator=(const TestVectors&) = delete;

  void ResetC() {
    for (size_t i = 0; i < C.size(); ++i) {
      C[i] = 0;
    }
  }

  const size_t M;
  const size_t N;
  std::vector<T> A;
  std::vector<T> B;
  std::vector<T> C;
};

//
// Reference vector dot product.
//
template <typename T>
void ReferenceVectorDotProd(TestVectors<T>& vectors) {
  for (size_t i = 0; i < vectors.N; ++i) {
    T sum = 0;
    for (size_t j = 0; j < vectors.M; ++j) {
      T cur_A = vectors.A[j];
      T cur_B = vectors.B[i + (vectors.N * j)];
      sum += cur_A * cur_B;
    }
    vectors.C[i] = sum;
  }
}

template <typename T>
class MlasVectorDotProductTest : public MlasTestBase {
 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("VectorDotProduct");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    // Smaller vectors
    ValidateUnpacked(TestVectors<T>(/*M=*/8, /*N=*/4));
    ValidateUnpacked(TestVectors<T>(/*M=*/4, /*N=*/8));
    ValidateUnpacked(TestVectors<T>(/*M=*/3, /*N=*/9));
    ValidateUnpacked(TestVectors<T>(/*M=*/9, /*N=*/3));

    // Medium vectors
    ValidateUnpacked(TestVectors<T>(/*M=*/22, /*N=*/32));
    ValidateUnpacked(TestVectors<T>(/*M=*/21, /*N=*/31));

    // Long vectors:
    ValidateUnpacked(TestVectors<T>(/*M=*/768, /*N=*/3072));
    ValidateUnpacked(TestVectors<T>(/*M=*/761, /*N=*/3011));
  }

 private:
  void ValidateUnpacked(TestVectors<T> vectors) {
    ReferenceVectorDotProd(vectors);
    std::vector<float> ref_C = vectors.C;
    vectors.ResetC();

    MlasVectorDotProduct(vectors.A.data(),
                         vectors.B.data(),
                         vectors.C.data(),
                         vectors.M,
                         vectors.N);

    for (size_t i = 0; i < vectors.N; ++i) {
      if (vectors.C[i] != ref_C[i]) {
        ASSERT_NEAR(vectors.C[i], ref_C[i], 1e-6f);
      }
    }
  }
};

template <>
MlasVectorDotProductTest<float>* MlasTestFixture<MlasVectorDotProductTest<float>>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute
             ? MlasDirectShortExecuteTests<MlasVectorDotProductTest<float>>::RegisterShortExecute()
             : 0;
});
