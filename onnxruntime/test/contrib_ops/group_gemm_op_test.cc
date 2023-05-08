// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"

namespace onnxruntime {
namespace test {
namespace group_gemm {

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

  test_cases.push_back(
      {"test 4D",
       {2, 2, 2, 2},
       {2, 2, 2, 2},
       {2, 2, 2, 2},
       {2, 3, 6, 11, 46, 55, 66, 79, 154, 171, 190, 211, 326, 351, 378, 407}});

  test_cases.push_back(
      {"test 4D and broadcast",
       {1, 2, 3, 2},
       {3, 2, 2, 1},
       {3, 2, 3, 1},
       {1, 3, 5, 33, 43, 53, 5, 23, 41, 85, 111, 137, 9, 43, 77, 137, 179, 221}});

  return test_cases;
}

// [batch,N,M]->[batch,M,N]
template <typename T>
static void Transpose(const std::vector<T>& src, std::vector<T>& dst, size_t batch, size_t N, size_t M) {
  for (size_t b = 0; b < batch; b++) {
    for (size_t n = 0; n < N * M; n++) {
      dst[b * N * M + n] = src[b * N * M + M * (n % N) + n / N];
    }
  }
}

// [batch,N,M]->[N,batch,M]
template <typename T>
static void TransposeBatch(const std::vector<T>& src, std::vector<T>& dst, size_t batch, size_t N, size_t M) {
  for (size_t i = 0; i < batch * N; ++i) {
    size_t src_pos = ((i % batch) * N + i / batch) * M;
    size_t dst_pos = i * M;
    for (size_t j = 0; j < M; j++) {
      dst[dst_pos + j] = src[src_pos + j];
    }
  }
}

template <typename T>
static void TransposeInput(const std::vector<T>& src, std::vector<T>& dst, const std::vector<int64_t>& dims,
                           std::vector<int64_t>& new_dims, bool is_trans, bool is_trans_batch) {
  if (dims.size() < 2 || (!is_trans && !is_trans_batch)) return;
  ORT_ENFORCE(!is_trans_batch || dims.size() >= 3);
  size_t batch = 1;
  size_t size = dims.size();
  new_dims.resize(size);
  for (size_t i = 0; i < size - 2; ++i) {
    batch *= static_cast<size_t>(dims[i]);
    new_dims[i + (is_trans_batch ? 1 : 0)] = dims[i];
  }
  size_t N = static_cast<size_t>(dims[size - 2]);
  size_t M = static_cast<size_t>(dims[size - 1]);
  if (is_trans && !is_trans_batch) {
    new_dims[size - 1] = dims[size - 2];
    new_dims[size - 2] = dims[size - 1];
    Transpose<T>(src, dst, batch, N, M);
  } else if (!is_trans && is_trans_batch) {
    new_dims[0] = dims[size - 2];
    TransposeBatch<T>(src, dst, batch, N, M);
  } else {
    new_dims[size - 1] = dims[size - 2];
    new_dims[0] = dims[size - 1];
    std::vector<T> intermediate(src.size());
    Transpose<T>(src, intermediate, batch, N, M);       // [batch,N,M]->[batch,M,N]
    TransposeBatch<T>(intermediate, dst, batch, M, N);  // [batch,M,N]->[M,batch,N]
  }
}

template <typename T>
void ProcessInputs(const std::vector<int64_t>& input_dims, const std::vector<T>& common_input_vals, bool is_trans,
                   bool is_trans_batch, std::vector<int64_t>& modified_input_dims, std::vector<T>& input_vals) {
  auto rank = input_dims.size();
  ORT_ENFORCE(rank >= 1);
  int64_t size0 = TensorShape::FromExistingBuffer(input_dims).SizeHelper(0, rank);
  std::vector<T> input_vals_raw(common_input_vals.cbegin(), common_input_vals.cbegin() + size0);
  input_vals = input_vals_raw;
  TransposeInput<T>(input_vals_raw, input_vals, input_dims, modified_input_dims, is_trans, is_trans_batch);
}

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
TEST(FusedMatMulOpTest, Float16_NoTranspose) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  std::vector<float> common_input_vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  for (auto t : GenerateSimpleTestCases<float>()) {
    OpTester test("FusedMatMul", 1, onnxruntime::kMSDomain);

    std::vector<int64_t> input0_dims(t.input0_dims);
    std::vector<float> input0_vals;
    ProcessInputs(t.input0_dims, common_input_vals, false, false, input0_dims, input0_vals);

    std::vector<int64_t> input1_dims(t.input1_dims);
    std::vector<float> input1_vals;
    ProcessInputs(t.input1_dims, common_input_vals, false, false, input1_dims, input1_vals);

    std::vector<MLFloat16> f_A(input0_vals.size());
    std::vector<MLFloat16> f_B(input1_vals.size());
    std::vector<MLFloat16> f_Y(t.expected_vals.size());
    ConvertFloatToMLFloat16(input0_vals.data(), f_A.data(), (int)input0_vals.size());
    ConvertFloatToMLFloat16(input1_vals.data(), f_B.data(), (int)input1_vals.size());
    ConvertFloatToMLFloat16(t.expected_vals.data(), f_Y.data(), (int)t.expected_vals.size());

    test.AddInput<MLFloat16>("A", input0_dims, f_A);
    test.AddInput<MLFloat16>("B", input1_dims, f_B, false);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("transBatchA", (int64_t)0);
    test.AddAttribute("transBatchB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);

    test.AddOutput<MLFloat16>("Y", t.expected_dims, f_Y);

    // Disable TensorRT because of unsupported data type
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }
}
#endif

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DNNL)
TEST(FusedMatMulOpTest, BFloat16_NoTranspose) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif

  std::vector<float> common_input_vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  for (auto t : GenerateSimpleTestCases<float>()) {
#if defined(USE_DNNL)
    // disable scalar or 1D tensor input for oneDNN EP.
    if (t.name == "test left 1D" ||
        t.name == "test right 1D" ||
        t.name == "test scalar output" ||
        t.name == "test 2D with empty input") {
      continue;
    }
#endif  //  USE_DNNL

    OpTester test("FusedMatMul", 1, onnxruntime::kMSDomain);

    std::vector<int64_t> input0_dims(t.input0_dims);
    std::vector<float> input0_vals;
    ProcessInputs(t.input0_dims, common_input_vals, false, false, input0_dims, input0_vals);

    std::vector<int64_t> input1_dims(t.input1_dims);
    std::vector<float> input1_vals;
    ProcessInputs(t.input1_dims, common_input_vals, false, false, input1_dims, input1_vals);

    std::vector<BFloat16> f_A = FloatsToBFloat16s(input0_vals);
    std::vector<BFloat16> f_B = FloatsToBFloat16s(input1_vals);
    std::vector<BFloat16> f_Y = FloatsToBFloat16s(t.expected_vals);

    test.AddInput<BFloat16>("A", input0_dims, f_A);
    test.AddInput<BFloat16>("B", input1_dims, f_B, false);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("transBatchA", (int64_t)0);
    test.AddAttribute("transBatchB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);

    test.AddOutput<BFloat16>("Y", t.expected_dims, f_Y);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
    execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
    execution_providers.push_back(DefaultRocmExecutionProvider());
#elif USE_DNNL
    execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}
#endif  //  USE_CUDA USE_RCOM USE_DNNL

}  // namespace group_gemm
}  // namespace test
}  // namespace onnxruntime
