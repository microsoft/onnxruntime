// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ctime>
#include <cstdlib>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"

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
void RunTest(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& indices_dims, bool has_axis = false,
             int64_t axis = 0LL) {
  size_t input_size =
      static_cast<size_t>(std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>()));
  size_t indices_size =
      static_cast<size_t>(std::accumulate(indices_dims.begin(), indices_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> dY_data = onnxruntime::test::ValueRange<T>(indices_size, static_cast<T>(1.0f), static_cast<T>(1.0f));
  size_t rank = input_dims.size();
  std::vector<int64_t> input_strides(rank);
  std::vector<int64_t> indices_strides(rank);
  input_strides[rank - 1] = indices_strides[rank - 1] = 1;
  if (rank > 1) {
    for (size_t i = rank - 1; i > 0; --i) {
      input_strides[i - 1] = input_dims[i] * input_strides[i];
      indices_strides[i - 1] = indices_dims[i] * indices_strides[i];
    }
  }

  int64_t new_axis = axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
  std::vector<TIndex> indices_data(indices_size);
  std::vector<T> dX_data(input_size, static_cast<T>(0.f));
  std::srand(static_cast<unsigned>(std::time(0)));
  for (size_t i = 0; i < indices_size; ++i) {
    // Negative index possible.
    int64_t index = (static_cast<int64_t>(std::rand()) % (input_dims[new_axis] * 2)) - input_dims[new_axis];
    indices_data[i] = static_cast<TIndex>(index);
    int64_t offset = 0;
    int64_t remain = static_cast<int64_t>(i);
    for (size_t j = 0; j < rank; ++j) {
      int64_t pos = static_cast<int64_t>(j) == new_axis ? (index < 0 ? index + input_dims[new_axis] : index)
                                                        : (remain / indices_strides[j]);
      offset += pos * input_strides[j];
      remain = remain % indices_strides[j];
    }
    Add(&dX_data[0] + offset, &dY_data[0] + i);
  }

  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  if (has_axis) test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<T>("dY", indices_dims, dY_data);
  test.AddInput<int64_t>("data_shape", {static_cast<int64_t>(rank)}, input_dims);
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

  // ([2,2,3,3,2],[2,2,3,2,2],axis=0) coalesce to ([2,6,3,2],[2,6,3,2],axis=0)
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

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
