// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class TestVectors {
 public:
  TestVectors(const size_t c_length, const size_t stride_length,
              bool small_values = false)
      : c_length(c_length), stride_length(stride_length) {
    for (size_t i = 0; i < stride_length; ++i) {
      if (small_values) {
        a.push_back(0.01f * (i + 1));
      } else {
        a.push_back((i + 1) * 1.0f);
      }
    }

    for (size_t i = 0; i < c_length; ++i) {
      b.insert(b.end(), a.begin(), a.end());
    }

    b_packed.resize(b.size());

    c.resize(c_length);
  }

  void ResetC() {
    for (size_t i = 0; i < c.size(); ++i) {
      c[i] = 0.0f;
    }
  }

  TestVectors() = default;
  TestVectors(const TestVectors&) = delete;
  TestVectors& operator=(const TestVectors&) = delete;

  const size_t c_length;
  const size_t stride_length;
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> b_packed;
  std::vector<float> c;
};


//
// Reference vector dot product.
// TODO(kreeger): add more documentation here.
//
template <typename T>
void ReferenceVectorDotProd(const T* a,
                            const T* b,
                            T* output,
                            size_t M,
                            size_t N) {
  for (size_t i = 0; i < N; ++i) {
    T sum = 0;
    for (size_t j = 0; j < M; ++j) {
      sum += a[j] * b[i + (N * j)];
    }
    output[i] = sum;
  }
}

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
    std::cerr << "Hi from MlasVectorDotProdTest!" << std::endl;

    TestVectors vectors(/*c_length=*/32, /*stride_length=*/16);

    MlasVectorDotProduct(vectors.a.data(),
                         vectors.b.data(),
                         vectors.c.data(),
                         vectors.c_length,
                         vectors.stride_length);

    TestVectors vectors_ref(/*c_length=*/32, /*stride_length=*/16);
    ReferenceVectorDotProd(vectors_ref.a.data(),
                           vectors_ref.b.data(),
                           vectors_ref.c.data(),
                           vectors_ref.c_length,
                           vectors_ref.stride_length);

    for (size_t i = 0; i < vectors.c_length; ++i) {
      ASSERT_EQ(vectors.c[i], vectors_ref.c[i]);
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

