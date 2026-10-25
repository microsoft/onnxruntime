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

TEST(ContribOpGatedAddTest, ProjectedGateBFloat16) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  for (bool sigmoid_in_fp32 : {false, true}) {
    for (int64_t width : {8, 2560}) {
      constexpr int64_t kRows = 4;
      constexpr int64_t kInputWidth = 16;
      std::vector<BFloat16> x(kRows * width);
      std::vector<BFloat16> y(kRows * width);
      std::vector<BFloat16> act(kRows * kInputWidth);
      std::vector<BFloat16> weight(kInputWidth);
      std::vector<BFloat16> expected(kRows * width);
      for (int64_t k = 0; k < kInputWidth; ++k) {
        weight[k] = BFloat16(static_cast<float>((k % 5) - 2) * 0.25f);
      }
      for (int64_t r = 0; r < kRows; ++r) {
        float dot = 0.0f;
        for (int64_t k = 0; k < kInputWidth; ++k) {
          act[r * kInputWidth + k] = BFloat16(static_cast<float>(((r + k) % 7) - 3) * 0.3f);
          dot = std::fma(act[r * kInputWidth + k].ToFloat(), weight[k].ToFloat(), dot);
        }
        const float gate = BFloat16(dot).ToFloat();
        const float exp_value = std::exp(gate >= 0.0f ? -gate : gate);
        const float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + exp_value) : exp_value / (1.0f + exp_value);
        const BFloat16 denominator(1.0f + BFloat16(exp_value).ToFloat());
        const BFloat16 reciprocal(1.0f / denominator.ToFloat());
        const BFloat16 rounded_sigmoid =
            sigmoid_in_fp32 ? BFloat16(sigmoid)
                            : (gate > 0.0f ? reciprocal : BFloat16(1.0f - reciprocal.ToFloat()));
        for (int64_t c = 0; c < width; ++c) {
          const int64_t idx = r * width + c;
          x[idx] = BFloat16(static_cast<float>((idx % 17) - 8) * 0.13f);
          y[idx] = BFloat16(static_cast<float>((idx % 11) - 5) * 0.37f);
          const BFloat16 product(y[idx].ToFloat() * rounded_sigmoid.ToFloat());
          expected[idx] = BFloat16(x[idx].ToFloat() + product.ToFloat());
        }
      }
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      if (auto cuda_ep = DefaultCudaExecutionProvider()) {
        providers.push_back(std::move(cuda_ep));
      }
      if (providers.empty()) GTEST_SKIP();
      OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
      tester.AddAttribute<int64_t>("sigmoid_in_fp32", sigmoid_in_fp32 ? 1 : 0);
      tester.AddInput<BFloat16>("X", {1, kRows, width}, x);
      tester.AddInput<BFloat16>("Y", {1, kRows, width}, y);
      tester.AddInput<BFloat16>("gate", {1, kRows, kInputWidth}, act);
      tester.AddInput<BFloat16>("gate_weight", {kInputWidth, 1}, weight);
      tester.AddOutput<BFloat16>("output", {1, kRows, width}, expected);
      tester.SetOutputTolerance(0.0f, 0.0f);
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
}

TEST(ContribOpGatedAddTest, ArgMaxBFloat16) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  for (bool keepdims : {false, true}) {
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    if (auto cuda_ep = DefaultCudaExecutionProvider()) {
      providers.push_back(std::move(cuda_ep));
    }
    if (providers.empty()) GTEST_SKIP();
    OpTester test("ArgMax", 13);
    test.AddAttribute<int64_t>("axis", -1);
    test.AddAttribute<int64_t>("keepdims", keepdims ? 1 : 0);
    test.AddInput<BFloat16>("logits", {2, 4},
                            ToTensorType<BFloat16>({1.0f, 5.0f, 5.0f, -1.0f, 4.0f, 3.0f, 2.0f, 1.0f}));
    test.AddOutput<int64_t>("indices", keepdims ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{2}, {1, 0});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

}  // namespace test
}  // namespace onnxruntime
