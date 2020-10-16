// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

namespace {
std::vector<MLFloat16> FloatToMLFloat16(const std::vector<float>& float_data) {
  std::vector<MLFloat16> new_data;
  for (const auto& f : float_data) {
    new_data.push_back(MLFloat16(math::floatToHalf(f)));
  }
  return new_data;
}

template <typename T>
std::vector<T> CalculateOutput(
    int64_t axis,
    const TensorShape& output_shape,
    const std::vector<T>& grad,
    const std::vector<int64_t>& indices) {
  const int64_t num_batches = output_shape.SizeToDimension(axis);
  const int64_t gather_dimension_size = output_shape[axis];
  const int64_t num_gathered_per_index = output_shape.SizeFromDimension(axis + 1);
  std::vector<T> output(output_shape.Size());
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto output_batch_offset =
        batch_idx * gather_dimension_size * num_gathered_per_index;
    const auto grad_batch_offset =
        batch_idx * indices.size() * num_gathered_per_index;
    for (size_t i = 0; i < indices.size(); ++i) {
      const auto grad_row_offset =
          grad_batch_offset + i * num_gathered_per_index;
      const auto output_row_offset =
          output_batch_offset + indices[i] * num_gathered_per_index;
      for (int64_t j = 0; j < num_gathered_per_index; ++j) {
        output[output_row_offset + j] += grad[grad_row_offset + j];
      }
    }
  }
  return output;
}

template <typename T>
void TestGatherGradWithRandomData(
    int64_t axis,
    const TensorShape& X_shape,
    const TensorShape& indices_shape) {
  ASSERT_LE(0, axis);
  ASSERT_LT(axis, X_shape.NumDimensions());

  const TensorShape dY_shape = [&]() {
    std::vector<int64_t> dY_dims = X_shape.GetDims();
    auto it = dY_dims.erase(dY_dims.begin() + axis);
    dY_dims.insert(
        it, indices_shape.GetDims().begin(), indices_shape.GetDims().end());
    return TensorShape(dY_dims);
  }();

  RandomValueGenerator random{};
  const auto grad = random.Uniform<T>(dY_shape.GetDims(), T{1}, T{10});
  const auto indices = random.Uniform<int64_t>(indices_shape.GetDims(), 0, X_shape[axis]);
  const auto output = CalculateOutput(axis, X_shape, grad, indices);

  OpTester test("GatherGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<int64_t>(
      "shape", {static_cast<int64_t>(X_shape.NumDimensions())}, X_shape.GetDims());
  test.AddInput<int64_t>("indices", indices_shape.GetDims(), indices);
  test.AddInput<T>("grad", dY_shape.GetDims(), grad);
  test.AddOutput<T>("output", X_shape.GetDims(), output);
  test.Run();
}
}  // namespace

#ifdef USE_CUDA
//TODO: Currently this cannot pass CI, due to GPU architecture problem
TEST(GatherOpTest, Gather_axis0_indices2d_half) {
  if (NeedSkipIfCudaArchLowerThan(700)) {
    return;
  }

  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<MLFloat16>("data", {3, 3},
                           FloatToMLFloat16({0.0f, 0.1f, 0.2f,
                                             1.0f, 1.1f, 1.2f,
                                             2.0f, 2.1f, 2.2f}));
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {1LL, 0LL,
                          2LL, 1LL});
  test.AddOutput<MLFloat16>("output", {2, 2, 3},
                            FloatToMLFloat16({1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f,
                                              2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f}));
  test.Run();
}

TEST(GatherGradOpTest, GatherGrad_axis0_indices2d_half) {
  if (NeedSkipIfCudaArchLowerThan(700)) {
    return;
  }

  OpTester test("GatherGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         {3, 3});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {0LL, 1LL,
                          0LL, 1LL});

  test.AddInput<MLFloat16>("grad", {2, 2, 3},
                           FloatToMLFloat16({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
  test.AddOutput<MLFloat16>("output", {3, 3},
                            FloatToMLFloat16({0, 2, 4, 6, 8, 10, 0, 0, 0}));
  test.Run();
}
#endif

TEST(GatherGradOpTest, GatherGrad_axis0_indices2d_float) {
  OpTester test("GatherGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         {3, 3});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {0LL, 1LL,
                          0LL, 1LL});

  test.AddInput<float>("grad", {2, 2, 3},
                       {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
  test.AddOutput<float>("output", {3, 3},
                        {0, 2, 4, 6, 8, 10, 0, 0, 0});
  test.Run();
}

TEST(GatherGradOpTest, Gather_axis1_float_impl2) {
  TestGatherGradWithRandomData<float>(1, {3, 4}, {6, 128});
}

TEST(GatherGradOpTest, Gather_axis0_float_impl2) {
  TestGatherGradWithRandomData<float>(0, {3, 4}, {6, 128});
}

TEST(GatherGradOpTest, GatherFewDistinctIndices) {
  TestGatherGradWithRandomData<float>(0, {2, 32}, {6, 128});
}

}  // namespace test
}  // namespace onnxruntime
