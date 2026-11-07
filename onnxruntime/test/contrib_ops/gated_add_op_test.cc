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

TEST(ContribOpGatedAddTest, Float16NonVectorizedHiddenSize) {
  RunGatedAddTest<MLFloat16>({3, 5, 7});
}

TEST(ContribOpGatedAddTest, BFloat16NonVectorizedHiddenSize) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  RunGatedAddTest<BFloat16>({3, 5, 7});
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
  struct ProjectionCase {
    int64_t width;
    int64_t input_width;
  };
  for (const ProjectionCase& test_case : {ProjectionCase{8, 16}, ProjectionCase{2560, 16}, ProjectionCase{2560, 2560},
                                          ProjectionCase{259, 300}}) {
    constexpr int64_t kRows = 4;
    const int64_t width = test_case.width;
    const int64_t input_width = test_case.input_width;
    std::vector<BFloat16> x(kRows * width);
    std::vector<BFloat16> y(kRows * width);
    std::vector<BFloat16> act(kRows * input_width);
    std::vector<BFloat16> weight(input_width);
    std::vector<BFloat16> expected(kRows * width);
    for (int64_t k = 0; k < input_width; ++k) {
      weight[k] = BFloat16(static_cast<float>((k % 5) - 2) * 0.25f);
    }
    for (int64_t r = 0; r < kRows; ++r) {
      float dot = 0.0f;
      for (int64_t k = 0; k < input_width; ++k) {
        act[r * input_width + k] = BFloat16(static_cast<float>(((r + k) % 7) - 3) * 0.125f);
        dot = std::fma(act[r * input_width + k].ToFloat(), weight[k].ToFloat(), dot);
      }
      const float gate = BFloat16(dot).ToFloat();
      const float exp_value = std::exp(gate >= 0.0f ? -gate : gate);
      const float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + exp_value) : exp_value / (1.0f + exp_value);
      const BFloat16 rounded_sigmoid(sigmoid);
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
    tester.AddInput<BFloat16>("X", {1, kRows, width}, x);
    tester.AddInput<BFloat16>("Y", {1, kRows, width}, y);
    tester.AddInput<BFloat16>("gate", {1, kRows, input_width}, act);
    tester.AddInput<BFloat16>("gate_weight", {input_width, 1}, weight);
    tester.AddOutput<BFloat16>("output", {1, kRows, width}, expected);
    tester.SetOutputTolerance(0.0f, 0.0f);
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(ContribOpGatedAddTest, ProjectedGateK1Regression) {
  for (auto& ep : AvailableGatedOpExecutionProviders()) {
    SCOPED_TRACE("EP (float): " + ep->Type());
    OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
    tester.AddInput<float>("X", {1, 1}, {0.0f});
    tester.AddInput<float>("Y", {1, 1}, {2.0f});
    tester.AddInput<float>("gate", {1, 1}, {0.0f});
    tester.AddInput<float>("gate_weight", {1, 1}, {1.0f});
    tester.AddOutput<float>("output", {1, 1}, {1.0f});

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectFailure, "GatedAdd gate_weight currently supports BFloat16 only",
               {}, nullptr, &providers);
  }

  for (auto& ep : AvailableGatedOpExecutionProviders()) {
    SCOPED_TRACE("EP (MLFloat16): " + ep->Type());
    OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
    tester.AddInput<MLFloat16>("X", {1, 1}, ToTensorType<MLFloat16>({0.0f}));
    tester.AddInput<MLFloat16>("Y", {1, 1}, ToTensorType<MLFloat16>({2.0f}));
    tester.AddInput<MLFloat16>("gate", {1, 1}, ToTensorType<MLFloat16>({0.0f}));
    tester.AddInput<MLFloat16>("gate_weight", {1, 1}, ToTensorType<MLFloat16>({1.0f}));
    tester.AddOutput<MLFloat16>("output", {1, 1}, ToTensorType<MLFloat16>({1.0f}));

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectFailure, "GatedAdd gate_weight currently supports BFloat16 only",
               {}, nullptr, &providers);
  }

  if (CudaHasBF16Support()) {
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    if (auto cuda_ep = DefaultCudaExecutionProvider()) {
      providers.push_back(std::move(cuda_ep));
    }
    if (!providers.empty()) {
      OpTester tester("GatedAdd", 1, onnxruntime::kMSDomain);
      tester.AddInput<BFloat16>("X", {1, 1}, ToTensorType<BFloat16>({0.0f}));
      tester.AddInput<BFloat16>("Y", {1, 1}, ToTensorType<BFloat16>({2.0f}));
      tester.AddInput<BFloat16>("gate", {1, 1}, ToTensorType<BFloat16>({0.0f}));
      tester.AddInput<BFloat16>("gate_weight", {1, 1}, ToTensorType<BFloat16>({1.0f}));
      tester.AddOutput<BFloat16>("output", {1, 1}, ToTensorType<BFloat16>({1.0f}));
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
    for (const char* op : {"ArgMax", "ArgMin"}) {
      const bool is_max = std::string(op) == "ArgMax";
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      if (auto cuda_ep = DefaultCudaExecutionProvider()) {
        providers.push_back(std::move(cuda_ep));
      }
      if (providers.empty()) GTEST_SKIP();
      OpTester test(op, 13);
      test.AddAttribute<int64_t>("axis", -1);
      test.AddAttribute<int64_t>("keepdims", keepdims ? 1 : 0);
      test.AddInput<BFloat16>("logits", {2, 4},
                              ToTensorType<BFloat16>({1.0f, 5.0f, 5.0f, -1.0f, 4.0f, 3.0f, 2.0f, 1.0f}));
      test.AddOutput<int64_t>("indices", keepdims ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{2},
                              is_max ? std::vector<int64_t>{1, 0} : std::vector<int64_t>{3, 3});
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }

    for (int64_t cols : {int64_t{1024}, int64_t{40000}}) {
      constexpr int64_t kRows = 2;
      const std::vector<int64_t> max_indices{17, cols - 5};
      const std::vector<int64_t> min_indices{29, cols / 2};
      std::vector<float> logits(static_cast<size_t>(kRows * cols));
      for (int64_t r = 0; r < kRows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          logits[static_cast<size_t>(r * cols + c)] = static_cast<float>((c + r) % 31) * 0.5f;
        }
        logits[static_cast<size_t>(r * cols + max_indices[static_cast<size_t>(r)])] = 256.0f;
        logits[static_cast<size_t>(r * cols + cols - 1)] = 256.0f;
        logits[static_cast<size_t>(r * cols + min_indices[static_cast<size_t>(r)])] = -256.0f;
        logits[static_cast<size_t>(r * cols + cols - 2)] = -256.0f;
      }
      for (const char* op : {"ArgMax", "ArgMin"}) {
        const bool is_max = std::string(op) == "ArgMax";
        std::vector<std::unique_ptr<IExecutionProvider>> providers;
        if (auto cuda_ep = DefaultCudaExecutionProvider()) {
          providers.push_back(std::move(cuda_ep));
        }
        if (providers.empty()) GTEST_SKIP();
        OpTester test(op, 13);
        test.AddAttribute<int64_t>("axis", -1);
        test.AddAttribute<int64_t>("keepdims", keepdims ? 1 : 0);
        test.AddInput<BFloat16>("logits", {kRows, cols}, ToTensorType<BFloat16>(logits));
        test.AddOutput<int64_t>("indices", keepdims ? std::vector<int64_t>{kRows, 1} : std::vector<int64_t>{kRows},
                                is_max ? max_indices : min_indices);
        test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
      }
    }

    for (const char* op : {"ArgMax", "ArgMin"}) {
      const bool is_max = std::string(op) == "ArgMax";
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      if (auto cuda_ep = DefaultCudaExecutionProvider()) {
        providers.push_back(std::move(cuda_ep));
      }
      if (providers.empty()) GTEST_SKIP();
      OpTester test(op, 13);
      test.AddAttribute<int64_t>("axis", 1);
      test.AddAttribute<int64_t>("keepdims", keepdims ? 1 : 0);
      test.AddInput<BFloat16>("logits", {3, 2, 2},
                              ToTensorType<BFloat16>({1.0f, 4.0f,
                                                      3.0f, 2.0f,

                                                      8.0f, 5.0f,
                                                      7.0f, 6.0f,

                                                      9.0f, 12.0f,
                                                      11.0f, 10.0f}));
      test.AddOutput<int64_t>("indices", keepdims ? std::vector<int64_t>{3, 1, 2} : std::vector<int64_t>{3, 2},
                              is_max ? std::vector<int64_t>{1, 0, 0, 1, 1, 0}
                                     : std::vector<int64_t>{0, 1, 1, 0, 0, 1});
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
