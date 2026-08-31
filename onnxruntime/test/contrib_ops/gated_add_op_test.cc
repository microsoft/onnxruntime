// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

// Collects every EP that has a GatedAdd kernel available in this build, so the fused-op tests
// exercise all of them.
std::vector<std::unique_ptr<IExecutionProvider>> AvailableGatedOpExecutionProviders() {
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  if (auto cuda_ep = DefaultCudaExecutionProvider()) {
    eps.push_back(std::move(cuda_ep));
  }
  if (auto webgpu_ep = DefaultWebGpuExecutionProvider()) {
    eps.push_back(std::move(webgpu_ep));
  }
  return eps;
}

template <typename T>
float RoundToType(float value) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return MLFloat16(value).ToFloat();
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return BFloat16(value).ToFloat();
  } else {
    return value;
  }
}

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

// BFloat16 is currently CUDA-only; other types run on every available EP.
template <typename T>
std::vector<std::unique_ptr<IExecutionProvider>> ExecutionProvidersForType() {
  if constexpr (std::is_same_v<T, BFloat16>) {
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    if (auto cuda_ep = DefaultCudaExecutionProvider()) {
      eps.push_back(std::move(cuda_ep));
    }
    return eps;
  } else {
    return AvailableGatedOpExecutionProviders();
  }
}

template <typename T>
void RunGatedAddTest(const std::vector<int64_t>& input_dims) {
  ASSERT_FALSE(input_dims.empty());
  int64_t rows = 1;
  for (size_t axis = 0; axis + 1 < input_dims.size(); ++axis) {
    rows *= input_dims[axis];
  }
  const int64_t hidden_size = input_dims.back();
  const size_t count = static_cast<size_t>(rows * hidden_size);
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-3.0f, 3.0f);
  std::vector<float> x(count);
  std::vector<float> y(count);
  std::vector<float> gate(static_cast<size_t>(rows));
  for (float& value : x) value = distribution(generator);
  for (float& value : y) value = distribution(generator);
  for (float& value : gate) value = distribution(generator);

  std::vector<float> expected(count);
  for (size_t index = 0; index < count; ++index) {
    const float x_value = RoundToType<T>(x[index]);
    const float y_value = RoundToType<T>(y[index]);
    const float gate_value = RoundToType<T>(gate[index / hidden_size]);
    const float product = RoundToType<T>(y_value * gate_value);
    expected[index] = RoundToType<T>(x_value + product);
  }

  std::vector<int64_t> gate_dims = input_dims;
  gate_dims.back() = 1;

  auto execution_providers = ExecutionProvidersForType<T>();
  if (execution_providers.empty()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("EP: " + ep->Type());
    OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
    tester.AddInput<T>("X", input_dims, ToTensorType<T>(x));
    tester.AddInput<T>("Y", input_dims, ToTensorType<T>(y));
    tester.AddInput<T>("gate", gate_dims, ToTensorType<T>(gate));
    if constexpr (std::is_same_v<T, MLFloat16>) {
      // Allows valid WebGPU FP16 rounding differences near cancellation.
      tester.AddOutput<T>("output", input_dims, ToTensorType<T>(expected), false, 0.001f, 0.01f);
    } else {
      tester.AddOutput<T>("output", input_dims, ToTensorType<T>(expected));
    }

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

}  // namespace

TEST(ContribOpGatedAddTest, Float) {
  RunGatedAddTest<float>({2, 3, 7});
}

TEST(ContribOpGatedAddTest, RankOne) {
  RunGatedAddTest<float>({7});
}

TEST(ContribOpGatedAddTest, EmptyOuterDimension) {
  RunGatedAddTest<float>({0, 7});
}

TEST(ContribOpGatedAddTest, ZeroHiddenDimension) {
  auto execution_providers = AvailableGatedOpExecutionProviders();
  for (auto& ep : execution_providers) {
    SCOPED_TRACE("EP: " + ep->Type());
    OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
    tester.AddInput<float>("X", {2, 3, 0}, {});
    tester.AddInput<float>("Y", {2, 3, 0}, {});
    tester.AddInput<float>("gate", {2, 3, 1}, std::vector<float>(6));
    tester.AddOutput<float>("output", {2, 3, 0}, {});

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectFailure, "X last dimension must be positive",
               {}, nullptr, &providers);
  }
}

TEST(ContribOpGatedAddTest, Float16) {
  RunGatedAddTest<MLFloat16>({1, 4, 2048});
}

TEST(ContribOpGatedAddTest, BFloat16) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  RunGatedAddTest<BFloat16>({1, 4, 2048});
}

TEST(ContribOpGatedAddTest, MismatchedYShape) {
  auto execution_providers = AvailableGatedOpExecutionProviders();
  for (auto& ep : execution_providers) {
    SCOPED_TRACE("EP: " + ep->Type());
    OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
    tester.AddInput<float>("X", {1, 2, 3}, std::vector<float>(6));
    tester.AddInput<float>("Y", {1, 2, 4}, std::vector<float>(8));
    tester.AddInput<float>("gate", {1, 2, 1}, std::vector<float>(2));
    tester.AddOutput<float>("output", {1, 2, 3}, std::vector<float>(6));

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectFailure, "Y must have the same shape as X",
               {}, nullptr, &providers);
  }
}

}  // namespace test
}  // namespace onnxruntime
