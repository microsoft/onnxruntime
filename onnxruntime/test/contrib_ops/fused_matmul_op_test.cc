// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace transpose_matmul {

template <typename T>
struct MatMulTestData {
  std::string name;
  std::vector<int64_t> input0_dims;
  std::vector<int64_t> input1_dims;
  std::vector<int64_t> expected_dims;
  std::vector<T> expected_vals;
};

template <typename T>
std::vector<MatMulTestData<T>> GenerateSimpleTestCases() {
  std::vector<MatMulTestData<T>> test_cases;

  test_cases.push_back(
      {"test padding and broadcast",
       {3, 1, 1, 2},
       {2, 2, 2},
       {3, 2, 1, 2},
       {2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55}});

  test_cases.push_back(
      {"test padding and broadcast",
       {2, 3, 2},
       {3, 2, 2, 1},
       {3, 2, 3, 1},
       {1, 3, 5, 33, 43, 53, 5, 23, 41, 85, 111, 137, 9, 43, 77, 137, 179, 221}});

  test_cases.push_back(
      {"test left 1D",
       {2},
       {3, 2, 1},
       {3, 1},
       {1, 3, 5}});

  test_cases.push_back(
      {"test right 1D",
       {3, 1, 2},
       {2},
       {3, 1},
       {1, 3, 5}});

  test_cases.push_back(
      {"test scalar output",
       {3},
       {3},
       {},
       {5}});

  test_cases.push_back(
      {"test 2D",
       {3, 4},
       {4, 3},
       {3, 3},
       {42, 48, 54, 114, 136, 158, 186, 224, 262}});

  test_cases.push_back(
      {"test 2D special",
       {2, 2, 3},
       {3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}});

  test_cases.push_back(
      {"test 2D special 2",
       {2, 2, 3},
       {1, 3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}});

  test_cases.push_back(
      {"test 2D special 3",
       {1, 2, 3},
       {1, 3, 4},
       {1, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92}});

  test_cases.push_back(
      {"test 2D with empty input",
       {0, 3},
       {3, 4},
       {0, 4},
       {}});

  return test_cases;
}

/* Transpose the last two dimentions */
template <typename T>
static void Transpose(const std::vector<T>& src, std::vector<T>& dst, const int64_t batch, const int64_t N, const int64_t M) {
  for (int64_t b = 0; b < batch; b++) {
    for (int64_t n = 0; n < N * M; n++) {
      int64_t i = n / N;
      int64_t j = n % N;
      dst[b * N * M + n] = src[b * N * M + M * j + i];
    }
  }
}

template <typename T>
void ProcessInputs(const std::vector<int64_t>& input_dims, const std::vector<T>& common_input_vals, bool trans_flag,
                   std::vector<int64_t>& modified_input_dims, std::vector<T>& input_vals) {
  auto rank = input_dims.size();
  ORT_ENFORCE(rank >= 1);
  int64_t size0 = TensorShape::ReinterpretBaseType(input_dims).SizeHelper(0, rank);
  std::vector<T> input_vals_raw(common_input_vals.cbegin(), common_input_vals.cbegin() + size0);
  input_vals.resize(size0);

  // transpose on 1-d does not take any effect.
  if (rank == 1) {
    trans_flag = false;
  }

  if (trans_flag) {
    modified_input_dims[rank - 1] = input_dims[rank - 2];
    modified_input_dims[rank - 2] = input_dims[rank - 1];
    auto batch_size = TensorShape::ReinterpretBaseType(input_dims).SizeHelper(0, rank - 2);
    Transpose<T>(input_vals_raw, input_vals, batch_size, input_dims[rank - 2], input_dims[rank - 1]);
  } else {
    input_vals = input_vals_raw;
  }
}

template <typename T>
void RunFusedMatMulTest(const char* op_name, int32_t opset_version = 7, bool transa = false, bool transb = false, float alpha = 1.0f, bool is_b_constant = false) {
  std::vector<T> common_input_vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (auto t : GenerateSimpleTestCases<T>()) {
    OpTester test(op_name, opset_version, onnxruntime::kMSDomain);

    std::vector<int64_t> input0_dims(t.input0_dims);
    std::vector<T> input0_vals;
    ProcessInputs(t.input0_dims, common_input_vals, transa, input0_dims, input0_vals);

    std::vector<int64_t> input1_dims(t.input1_dims);
    std::vector<T> input1_vals;
    ProcessInputs(t.input1_dims, common_input_vals, transb, input1_dims, input1_vals);

    test.AddInput<T>("A", input0_dims, input0_vals);

    test.AddInput<T>("B", input1_dims, input1_vals, is_b_constant);

    test.AddAttribute("transA", (int64_t)transa);
    test.AddAttribute("transB", (int64_t)transb);
    test.AddAttribute("alpha", alpha);

    if (alpha != 1.0f) {
      std::transform(
          t.expected_vals.begin(), t.expected_vals.end(), t.expected_vals.begin(),
          [alpha](const T& val) -> T { return alpha * val; });
    }

    test.AddOutput<T>("Y", t.expected_dims, t.expected_vals);

    // Disable TensorRT because of unsupported data type
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }
}

TEST(FusedMatMulOpTest, FloatTypeNoTranspose) {
  RunFusedMatMulTest<float>("FusedMatMul", 1);
}

#if defined(USE_CUDA) || defined(USE_ROCM)  // double support only implemented in CUDA/ROCM kernel

TEST(FusedMatMulOpTest, DoubleTypeNoTranspose) {
  RunFusedMatMulTest<double>("FusedMatMul", 1);
}
#endif

TEST(FusedMatMulOpTest, FloatTypeTransposeA) {
  RunFusedMatMulTest<float>("FusedMatMul", 1, true, false);
}

TEST(FusedMatMulOpTest, FloatTypeTransposeB) {
  RunFusedMatMulTest<float>("FusedMatMul", 1, false, true);
  // b is constant. This tests weight packing logic
  RunFusedMatMulTest<float>("FusedMatMul", 1, false, true, 1.0f, true);
}

TEST(FusedMatMulOpTest, FloatTypeTransposeAB) {
  RunFusedMatMulTest<float>("FusedMatMul", 1, true, true);

  // b is constant. This tests weight packing logic
  RunFusedMatMulTest<float>("FusedMatMul", 1, true, true, 1.0f, true);
}

TEST(FusedMatMulOpTest, FloatTypeScale) {
  RunFusedMatMulTest<float>("FusedMatMul", 1, false, false, 0.5f);
  RunFusedMatMulTest<float>("FusedMatMul", 1, true, false, 2.0f);
  RunFusedMatMulTest<float>("FusedMatMul", 1, true, true, 4.0f);

  // now run tests with b constant.
  RunFusedMatMulTest<float>("FusedMatMul", 1, false, false, 0.5f, true);
  RunFusedMatMulTest<float>("FusedMatMul", 1, true, false, 2.0f, true);
  RunFusedMatMulTest<float>("FusedMatMul", 1, true, true, 4.0f, true);
}

}  // namespace transpose_matmul
}  // namespace test
}  // namespace onnxruntime
