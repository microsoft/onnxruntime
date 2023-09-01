// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/common/optional.h"
#include "core/util/math.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<T> CalculateOutput(int64_t axis,
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
void ConfigureGatherGradRandomDataOpTester(int64_t axis,
                                           const TensorShape& X_shape,
                                           const TensorShape& indices_shape,
                                           optional<RandomValueGenerator::RandomSeedType> random_seed,
                                           OpTester& test) {
  ASSERT_LE(0, axis);
  ASSERT_LT(static_cast<size_t>(axis), X_shape.NumDimensions());

  const TensorShape dY_shape = [&]() {
    TensorShapeVector dY_dims = X_shape.AsShapeVector();
    auto it = dY_dims.erase(dY_dims.begin() + axis);
    dY_dims.insert(
        it, indices_shape.GetDims().begin(), indices_shape.GetDims().end());
    return TensorShape(dY_dims);
  }();

  RandomValueGenerator random{random_seed};
  const auto grad = random.Uniform<T>(dY_shape.GetDims(), T{1}, T{10});
  const auto indices = random.Uniform<int64_t>(indices_shape.GetDims(), 0, X_shape[axis]);
  const auto output = CalculateOutput(axis, X_shape, grad, indices);

  test.AddAttribute<int64_t>("axis", axis);
  // auto shape_dims = X_shape.GetDims();
  // std::vector<int64_t> v_dims(shape_dims.cbegin(), shape_dims.cend());
  test.AddInput<int64_t>("shape", {static_cast<int64_t>(X_shape.NumDimensions())}, X_shape.GetDims());
  test.AddInput<int64_t>("indices", indices_shape.AsShapeVector(), indices);
  test.AddInput<T>("grad", dY_shape.AsShapeVector(), grad);
  test.AddOutput<T>("output", X_shape.AsShapeVector(), output);
}

template <typename T>
void ConfigureGatherGradRandomDataOpTester(
    int64_t axis,
    const TensorShape& X_shape,
    const TensorShape& indices_shape,
    OpTester& test) {
  ConfigureGatherGradRandomDataOpTester<T>(axis, X_shape, indices_shape, {}, test);
}

template <typename T>
void RunGatherGradTestWithRandomData(
    int64_t axis,
    const TensorShape& X_shape,
    const TensorShape& indices_shape,
    optional<float> absolute_error = {}) {
  OpTester test("GatherGrad", 1, kMSDomain);
  ConfigureGatherGradRandomDataOpTester<T>(axis, X_shape, indices_shape, test);
  if (absolute_error.has_value()) {
    test.SetOutputAbsErr("output", absolute_error.value());
  }
  test.Run();
}
}  // namespace

#if defined(USE_CUDA) || defined(USE_ROCM)
// TODO: Currently this cannot pass CI, due to GPU architecture problem
TEST(GatherOpTest, Gather_axis0_indices2d_half) {
#ifdef USE_CUDA
  if (NeedSkipIfCudaArchLowerThan(700)) {
    return;
  }
#endif
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<MLFloat16>("data", {3, 3},
                           FloatsToMLFloat16s({0.0f, 0.1f, 0.2f,
                                               1.0f, 1.1f, 1.2f,
                                               2.0f, 2.1f, 2.2f}));
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {1LL, 0LL,
                          2LL, 1LL});
  test.AddOutput<MLFloat16>("output", {2, 2, 3},
                            FloatsToMLFloat16s({1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f,
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
                           FloatsToMLFloat16s({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
  test.AddOutput<MLFloat16>("output", {3, 3},
                            FloatsToMLFloat16s({0, 2, 4, 6, 8, 10, 0, 0, 0}));
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

TEST(GatherGradOpTest, GatherGrad_negative_indices) {
  OpTester test("GatherGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         {3, 3});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {0LL, -1LL,
                          0LL, -2LL});

  test.AddInput<float>("grad", {2, 2, 3},
                       {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
  test.AddOutput<float>("output", {3, 3},
                        {0, 2, 4, 3, 4, 5, 3, 4, 5});
  test.Run();
}

TEST(GatherGradOpTest, Gather_axis1_float_impl2) {
  RunGatherGradTestWithRandomData<float>(1, {3, 4}, {6, 128});
}

TEST(GatherGradOpTest, Gather_axis0_float_impl2) {
  RunGatherGradTestWithRandomData<float>(0, {3, 4}, {6, 128});
}

TEST(GatherGradOpTest, GatherFewDistinctIndices) {
  // account for error from adding longer sequences of floats in different orders
  optional<float> absolute_error{5e-3f};
  RunGatherGradTestWithRandomData<float>(0, {2, 32}, {6, 128}, absolute_error);
}

#if defined(USE_CUDA) || defined(USE_ROCM)
namespace {
void RunGatherGradConsistentOutputTest(
    int64_t axis,
    const TensorShape& X_shape,
    const TensorShape& indices_shape) {
  const auto random_seed = static_cast<RandomValueGenerator::RandomSeedType>(GetTestRandomSeed());
  std::map<std::string, std::vector<std::vector<float>>> provider_outputs;
  for (int i = 0; i < 2; ++i) {
    OpTester test("GatherGrad", 1, kMSDomain);
    ConfigureGatherGradRandomDataOpTester<float>(axis, X_shape, indices_shape, random_seed, test);

    auto output_handler =
        [&provider_outputs](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
          ASSERT_EQ(fetches.size(), 1);
          const Tensor& output_tensor = fetches[0].Get<Tensor>();
          const auto output_size = output_tensor.Shape().Size();
          std::vector<float> output;
          output.reserve(output_size);
          std::copy_n(output_tensor.Data<float>(), output_size, std::back_inserter(output));
          provider_outputs[provider_type].emplace_back(std::move(output));
        };

    test.SetCustomOutputVerifier(output_handler);

    // current CPU implementation is non-deterministic
    const std::unordered_set<std::string> excluded_providers{kCpuExecutionProvider};

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
  }

  for (const auto& kvp : provider_outputs) {
    SCOPED_TRACE(kvp.first);
    const auto& outputs = kvp.second;
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(outputs[0], outputs[1]);
  }
}
}  // namespace

TEST(GatherGradOpTest, ConsistentOutput) {
  RunGatherGradConsistentOutputTest(0, {256 * 1024}, {1024 * 1024});
}

TEST(GatherGradOpTest, ConsistentOutputFewDistinctIndices) {
  RunGatherGradConsistentOutputTest(0, {2}, {1024 * 1024});
}

TEST(GatherGradOpTest, LargeGatherElementsPerIndex) {
  RunGatherGradConsistentOutputTest(0, {8, 256, 196, 192}, {4});
}
#endif

}  // namespace test
}  // namespace onnxruntime
