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

template <typename T>
void RunGatedAddTest(int batch_size, int sequence_length, int hidden_size) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  const int rows = batch_size * sequence_length;
  const size_t count = static_cast<size_t>(rows) * hidden_size;
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

  const std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  const std::vector<int64_t> gate_dims = {batch_size, sequence_length, 1};
  OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
  tester.AddInput<T>("X", input_dims, ToTensorType<T>(x));
  tester.AddInput<T>("Y", input_dims, ToTensorType<T>(y));
  tester.AddInput<T>("gate", gate_dims, ToTensorType<T>(gate));
  tester.AddOutput<T>("output", input_dims, ToTensorType<T>(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(ContribOpGatedAddTest, Float) {
  RunGatedAddTest<float>(2, 3, 7);
}

TEST(ContribOpGatedAddTest, Float16) {
  RunGatedAddTest<MLFloat16>(1, 4, 2048);
}

TEST(ContribOpGatedAddTest, BFloat16) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  RunGatedAddTest<BFloat16>(1, 4, 2048);
}

}  // namespace test
}  // namespace onnxruntime