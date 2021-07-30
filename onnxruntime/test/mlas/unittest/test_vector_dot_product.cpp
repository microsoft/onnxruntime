// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class TestVectors {
 public:
  TestVectors(const size_t M, const size_t N,
              bool small_values = false)
      : M(M), N(N), M_packed(0) {
    for (size_t i = 0; i < M; ++i) {
      if (small_values) {
        A.push_back(0.01f * (i + 1));
      } else {
        A.push_back((i + 1) * 1.0f);
      }
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

    // TODO(kreeger): determine pre-packed version of this ...
    // Ensure that |B_packed| is in increments of 16 per row:
    if (M % 16 != 0) {
      M_packed = 16 * ((M / 16) + 1);
    } else {
      M_packed = M;
    }
    B_packed.resize(N * M_packed);

    C.resize(N);
  }

  TestVectors(const TestVectors& copy) = delete;
  TestVectors() = default;
  TestVectors& operator=(const TestVectors&) = delete;

  void ResetC() {
    for (size_t i = 0; i < C.size(); ++i) {
      C[i] = 0.0f;
    }
  }

  const size_t M;
  const size_t N;
  size_t M_packed;
  std::vector<float> A;
  std::vector<float> B;
  std::vector<float> B_packed;
  std::vector<float> C;
};

//
// Reference vector dot product.
// TODO(kreeger): add more documentation here.
//
#pragma warning(disable: 4189)
void ReferenceVectorDotProd(TestVectors& vectors) {
  for (size_t i = 0; i < vectors.N; ++i) {
    float sum = 0;
    for (size_t j = 0; j < vectors.M; ++j) {
      sum += vectors.A[j] * vectors.B[i + (vectors.N * j)];
    }
    vectors.C[i] = sum;
  }
}
#pragma warning(default: 4189)

template <typename T, bool Packed>
class MlasVectorDotProdTest : public MlasTestBase {
  void Test() {
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("VectorDotProd");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    ExecuteSmall();
    ExecuteMedium();
    ExecuteLong();
  }

 private:
  void ExecuteSmall() {
    ValidateUnpacked(TestVectors(/*M=*/4, /*N=*/8));
    ValidateUnpacked(TestVectors(/*M=*/3, /*N=*/9));
  }

  void ExecuteMedium() {
    ValidateUnpacked(TestVectors(/*M=*/22, /*N=*/32));
    ValidateUnpacked(TestVectors(/*M=*/21, /*N=*/31));
  }

  void ExecuteLong() {
    ValidateUnpacked(TestVectors(/*M=*/768, /*N=*/3072, /*small_values=*/true));
    ValidateUnpacked(TestVectors(/*M=*/761, /*N=*/3011, /*small_values=*/true));
  }

  void ValidateUnpacked(TestVectors vectors) {
    ReferenceVectorDotProd(vectors);
    std::vector<float> ref_C = vectors.C;
    vectors.ResetC();

    // Consider making the regular method do the transpose.
    MlasTranspose(vectors.B.data(),
                  vectors.B_packed.data(),
                  vectors.M,
                  vectors.N);
    MlasVectorDotProduct(vectors.A.data(),
                         vectors.B_packed.data(),
                         vectors.C.data(),
                         vectors.M,
                         vectors.N);

    for (size_t i = 0; i < vectors.N; ++i) {
      ASSERT_NEAR(vectors.C[i], ref_C[i], 1e-2f);
    }
  }
};


template <>
MlasVectorDotProdTest<float, false>* MlasTestFixture<MlasVectorDotProdTest<float, false>>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute
             ? MlasDirectShortExecuteTests<MlasVectorDotProdTest<float, false>>::RegisterShortExecute()
             : 0;
});
