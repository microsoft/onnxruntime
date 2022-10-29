// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ctime>
#include <cstdlib>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(ENABLE_TRAINING) && (defined(USE_CUDA) || defined(USE_ROCM))
#include "test/providers/kernel_compute_test_utils.h"
#endif

namespace onnxruntime {
namespace cuda {
namespace test {

namespace {

template <typename T>
void Add(T* a, const T* b) {
  *a = *a + *b;
}

template <>
void Add<MLFloat16>(MLFloat16* a, const MLFloat16* b) {
  *a = MLFloat16((*a).ToFloat() + (*b).ToFloat());
}

template <typename T, typename TIndex>
void GetData(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& indices_dims,
             const std::vector<int64_t>& indices_strides, int64_t axis, std::vector<T>& dY_data,
             std::vector<TIndex>& indices_data, std::vector<T>& dX_data) {
  size_t dx_size = static_cast<size_t>(onnxruntime::test::detail::SizeFromDims(input_dims));
  size_t indices_size = static_cast<size_t>(onnxruntime::test::detail::SizeFromDims(indices_dims, indices_strides));
  bool is_strided_indices = !indices_strides.empty();
  size_t dy_size =
      is_strided_indices ? static_cast<size_t>(onnxruntime::test::detail::SizeFromDims(indices_dims)) : indices_size;
  dY_data = onnxruntime::test::ValueRange<T>(dy_size, static_cast<T>(1.0f), static_cast<T>(1.0f));
  size_t rank = input_dims.size();
  std::vector<int64_t> input_strides(rank);
  std::vector<int64_t> output_strides(rank);
  input_strides[rank - 1] = output_strides[rank - 1] = 1;
  if (rank > 1) {
    for (size_t i = rank - 1; i > 0; --i) {
      input_strides[i - 1] = input_dims[i] * input_strides[i];
      output_strides[i - 1] = indices_dims[i] * output_strides[i];
    }
  }

  indices_data.resize(indices_size);
  dX_data.resize(dx_size, static_cast<T>(0.f));
  std::srand(static_cast<unsigned>(std::time(0)));
  for (size_t i = 0; i < indices_size; ++i) {
    // Negative index possible.
    indices_data[i] =
        static_cast<TIndex>((static_cast<int64_t>(std::rand()) % (input_dims[axis] * 2)) - input_dims[axis]);
  }
  for (size_t i = 0; i < dy_size; ++i) {
    int64_t input_offset = 0;
    int64_t remain = static_cast<int64_t>(i);
    int64_t indices_offset = is_strided_indices ? 0 : remain;
    for (size_t j = 0; j < rank; ++j) {
      int64_t q = remain / output_strides[j];
      if (static_cast<int64_t>(j) != axis) input_offset += q * input_strides[j];
      if (is_strided_indices) indices_offset += q * indices_strides[j];
      remain = remain % output_strides[j];
    }
    int64_t index = static_cast<int64_t>(indices_data[indices_offset]);
    input_offset += (index < 0 ? index + input_dims[axis] : index) * input_strides[axis];
    Add(&dX_data[0] + input_offset, &dY_data[0] + i);
  }
}

template <typename T, typename TIndex>
void RunTest(std::initializer_list<int64_t> input_dims, std::initializer_list<int64_t> indices_dims,
             bool has_axis = false, int64_t axis = 0LL) {
  std::vector<T> dY_data;
  std::vector<TIndex> indices_data;
  std::vector<T> dX_data;
  int64_t new_axis = axis < 0 ? axis + static_cast<int64_t>(input_dims.size()) : axis;
  GetData(input_dims, indices_dims, {}, new_axis, dY_data, indices_data, dX_data);
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  if (has_axis) test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<T>("dY", indices_dims, dY_data);
  test.AddInput<int64_t>("data_shape", {static_cast<int64_t>(input_dims.size())}, input_dims);
  test.AddInput<TIndex>("indices", indices_dims, indices_data);
  test.AddOutput<T>("output", input_dims, dX_data);
  test.Run();
}

template <typename T>
void RunTestWrapper() {
  RunTest<T, int32_t>({2, 3}, {1, 2});
  RunTest<T, int32_t>({2, 2}, {2, 2}, true, 1LL);
  RunTest<T, int64_t>({2, 2}, {2, 2}, true, -1LL);
  RunTest<T, int32_t>({2, 2, 2}, {1, 2, 1}, true, 1LL);
  RunTest<T, int64_t>({2, 2, 2}, {1, 2, 1}, true, 2LL);
  RunTest<T, int32_t>({3, 3}, {3, 2}, true, 1LL);
  RunTest<T, int64_t>({3, 3}, {3, 2});
  RunTest<T, int32_t>({3}, {2}, true, 0LL);

  // ([2,2,2],[2,2,2],axis=0) coalesce to ([2,4],[2,4],axis=0)
  RunTest<T, int32_t>({2, 2, 2}, {2, 2, 2});

  // ([2,2,2],[3,2,2],axis=0) coalesce to ([2,4],[3,4],axis=0)
  RunTest<T, int64_t>({2, 2, 2}, {3, 2, 2}, true);

  // ([2,2,2,3],[3,2,2,2],axis=0) coalesce to ([2,4,3],[3,4,2],axis=0)
  RunTest<T, int32_t>({2, 2, 2, 3}, {3, 2, 2, 2}, true, 0LL);

  // ([2,2,3,3,2],[2,2,3,2,2],axis=0) coalesce to ([2,6,3,2],[2,6,2,2],axis=0)
  RunTest<T, int64_t>({2, 2, 3, 3, 2}, {2, 2, 3, 2, 2});

  // ([2,2,1,3,1],[2,2,1,2,1],axis=0) coalesce to ([2,2,3],[2,2,2],axis=0)
  RunTest<T, int32_t>({2, 2, 1, 3, 1}, {2, 2, 1, 2, 1});

  // No coalesce
  RunTest<T, int64_t>({2, 3, 2}, {2, 3, 2}, true, -2LL);

  // ([2,2,2],[2,2,3],axis=2) coalesce to ([4,2],[4,3],axis=1)
  RunTest<T, int64_t>({2, 2, 2}, {2, 2, 3}, true, 2LL);

  // ([2,2,3,2],[2,2,2,3],axis=-1) coalesce to ([4,3,2],[4,2,3],axis=2)
  RunTest<T, int32_t>({2, 2, 3, 2}, {2, 2, 2, 3}, true, -1LL);

  // ([2,1,2,3,1,2],[2,1,2,2,1,2],axis=5) coalesce to ([4,3,2],[4,2,2],axis=2)
  RunTest<T, int64_t>({2, 1, 2, 3, 1, 2}, {2, 1, 2, 2, 1, 2}, true, 5LL);

  // ([2,1,2,2,3,2,2],[2,1,2,3,2,2,2],axis=3) coalesce to ([4,2,3,4],[4,3,2,4],axis=1)
  RunTest<T, int32_t>({2, 1, 2, 2, 3, 2, 2}, {2, 1, 2, 3, 2, 2, 2}, true, 3LL);

  // ([2,1,1,2,3,2,3],[2,1,1,2,3,2,2],axis=-5) coalesce to ([2,1,12,3],[2,1,12,2],axis=1)
  RunTest<T, int64_t>({2, 1, 1, 2, 3, 2, 3}, {2, 1, 1, 2, 3, 2, 2}, true, -5LL);
}

#if defined(ENABLE_TRAINING) && (defined(USE_CUDA) || defined(USE_ROCM))
template <typename T, typename TIndex>
void RunKernelComputeTest(std::initializer_list<int64_t> input_dims, std::initializer_list<int64_t> indices_dims,
                          std::initializer_list<int64_t> indices_strides = {}, bool has_axis = false,
                          int64_t axis = 0LL) {
  std::vector<T> dY_data;
  std::vector<TIndex> indices_data;
  std::vector<T> dX_data;
  int64_t new_axis = axis < 0 ? axis + static_cast<int64_t>(input_dims.size()) : axis;
  GetData(input_dims, indices_dims, indices_strides, new_axis, dY_data, indices_data, dX_data);
#ifdef USE_CUDA
  const char* provider = kCudaExecutionProvider;
#else  // USE_ROCM
  const char* provider = kRocmExecutionProvider;
#endif
  onnxruntime::test::KernelComputeTester test("GatherElementsGrad", provider, 1, kMSDomain);
  if (has_axis) test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<T>("dY", indices_dims, dY_data);
  test.AddInput<int64_t>("data_shape", {static_cast<int64_t>(input_dims.size())}, input_dims, {}, true);
  test.AddInput<TIndex>("indices", indices_dims, indices_data, indices_strides);
  test.AddOutput<T>("output", input_dims, dX_data);
  test.Run();
}

template <typename T>
void RunKernelComputeTestWrapper() {
  // Contiguous indices.
  RunKernelComputeTest<T, int32_t>({2, 3}, {1, 2});

  // Strided indices.
  RunKernelComputeTest<T, int64_t>({3, 3}, {3, 2}, {1, 2});
  RunKernelComputeTest<T, int32_t>({3, 3}, {3, 2}, {2, 0}, true, 1LL);
  RunKernelComputeTest<T, int64_t>({3}, {2}, {0}, true, 0LL);

  // No coalesce.
  RunKernelComputeTest<T, int64_t>({2, 3, 2}, {2, 3, 2}, {6, 0, 1}, true, -1LL);
  RunKernelComputeTest<T, int32_t>({2, 3, 3, 3}, {2, 3, 3, 4}, {36, 0, 4, 1}, true, -1LL);

  // Coalesce to ([6,3,3],[6,3,4],strides=[12,0,1],axis=2).
  RunKernelComputeTest<T, int64_t>({2, 3, 3, 3}, {2, 3, 3, 4}, {36, 12, 0, 1}, true, 3LL);
}
#endif

}  // namespace

TEST(GatherElementsGrad, float) { RunTestWrapper<float>(); }

TEST(GatherElementsGrad, double) { RunTestWrapper<double>(); }

TEST(GatherElementsGrad, MLFloat16) { RunTestWrapper<MLFloat16>(); }

TEST(GatherElementsGrad, IndicesUpdatesDontMatch) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("dY", {1, 2}, {1.1f, 2.1f});
  std::vector<int64_t> data_shape = {1, 5};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);
  test.AddInput<int64_t>("indices", {1, 3}, {1, 3, 3});
  test.AddOutput<float>("dX", {1, 5}, {1.0f, 3.1f, 3.0f, 6.1f, 5.0f});
  test.Run(onnxruntime::test::OpTester::ExpectResult::kExpectFailure, "");
}

#if defined(ENABLE_TRAINING) && (defined(USE_CUDA) || defined(USE_ROCM))
TEST(GatherElementsGrad, Strided_float) { RunKernelComputeTestWrapper<float>(); }

TEST(GatherElementsGrad, Strided_double) { RunKernelComputeTestWrapper<double>(); }

TEST(GatherElementsGrad, Strided_MLFloat16) { RunKernelComputeTestWrapper<MLFloat16>(); }
#endif

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
