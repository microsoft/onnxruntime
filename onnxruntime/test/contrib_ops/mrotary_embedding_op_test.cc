// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

enum class TensorType {
  kFloat,
  kFloat16,
};

class WebGpuOpTester final : public OpTester {
 public:
  WebGpuOpTester() : OpTester("MRotaryEmbedding", 1, kMSDomain) {}

 private:
  void AddNodes(Graph& graph,
                std::vector<NodeArg*>& graph_input_defs,
                std::vector<NodeArg*>& graph_output_defs,
                std::vector<std::function<void(Node&)>>& add_attribute_funcs) override {
    OpTester::AddNodes(graph, graph_input_defs, graph_output_defs, add_attribute_funcs);
    for (auto& node : graph.Nodes()) {
      node.SetExecutionProviderType(kWebGpuExecutionProvider);
    }
  }
};

std::unique_ptr<OpTester> CreateMRotaryEmbeddingTester(const IExecutionProvider& execution_provider) {
  return execution_provider.Type() == kWebGpuExecutionProvider
             ? std::make_unique<WebGpuOpTester>()
             : std::make_unique<OpTester>("MRotaryEmbedding", 1, kMSDomain);
}

std::vector<std::unique_ptr<IExecutionProvider>> GetAvailableGpuExecutionProviders() {
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (HasCudaEnvironment(0)) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }
  if (auto webgpu_ep = DefaultWebGpuExecutionProvider()) {
    execution_providers.push_back(std::move(webgpu_ep));
  }
  return execution_providers;
}

std::unique_ptr<IExecutionProvider> CreateGpuExecutionProvider(const std::string& provider_type) {
  if (provider_type == kCudaExecutionProvider) {
    return DefaultCudaExecutionProvider();
  }
  if (provider_type == kWebGpuExecutionProvider) {
    return DefaultWebGpuExecutionProvider();
  }
  return nullptr;
}

TEST(ContribOpMRotaryEmbeddingTest, UnknownGpuExecutionProviderReturnsNull) {
  EXPECT_EQ(CreateGpuExecutionProvider("UnknownExecutionProvider"), nullptr);
}

void RunMRotaryEmbeddingTest(const std::vector<int64_t>& input_shape,
                             const std::vector<float>& input_data,
                             const std::vector<int64_t>& position_ids,
                             const std::vector<float>& cos_cache,
                             const std::vector<float>& sin_cache,
                             const std::vector<float>& expected_output,
                             const std::vector<int64_t>& mrope_section,
                             int64_t mrope_layout,
                             int64_t interleaved,
                             float scale,
                             int64_t num_heads,
                             int64_t rotary_embedding_dim,
                             TensorType tensor_type) {
  const std::vector<int64_t> position_ids_shape{3, input_shape[0], input_shape[input_shape.size() - 2]};
  const std::vector<int64_t> cache_shape{4, rotary_embedding_dim / 2};

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (HasCudaEnvironment(tensor_type == TensorType::kFloat16 ? 530 : 0)) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }
  execution_providers.push_back(DefaultCpuExecutionProvider());
  if (auto webgpu_ep = DefaultWebGpuExecutionProvider()) {
    execution_providers.push_back(std::move(webgpu_ep));
  }

  for (auto& ep : execution_providers) {
    auto test = CreateMRotaryEmbeddingTester(*ep);
    test->AddAttribute<int64_t>("num_heads", num_heads);
    test->AddAttribute<int64_t>("rotary_embedding_dim", rotary_embedding_dim);
    test->AddAttribute<int64_t>("mrope_layout", mrope_layout);
    test->AddAttribute<int64_t>("interleaved", interleaved);
    test->AddAttribute<float>("scale", scale);
    test->AddAttribute<std::vector<int64_t>>("mrope_section", mrope_section);

    if (tensor_type == TensorType::kFloat) {
      test->AddInput<float>("input", input_shape, input_data);
      test->AddInput<int64_t>("position_ids", position_ids_shape, position_ids);
      test->AddInput<float>("cos_cache", cache_shape, cos_cache);
      test->AddInput<float>("sin_cache", cache_shape, sin_cache);
      test->AddOutput<float>("output", input_shape, expected_output);
    } else {
      test->AddInput<MLFloat16>("input", input_shape, ToFloat16(input_data));
      test->AddInput<int64_t>("position_ids", position_ids_shape, position_ids);
      test->AddInput<MLFloat16>("cos_cache", cache_shape, ToFloat16(cos_cache));
      test->AddInput<MLFloat16>("sin_cache", cache_shape, ToFloat16(sin_cache));
      test->AddOutput<MLFloat16>("output", input_shape, ToFloat16(expected_output));
      test->SetOutputAbsErr("output", 0.01f);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test->Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
  }
}

void RunMRotaryEmbeddingTests(const std::vector<int64_t>& input_shape,
                              const std::vector<float>& input_data,
                              const std::vector<int64_t>& position_ids,
                              const std::vector<float>& cos_cache,
                              const std::vector<float>& sin_cache,
                              const std::vector<float>& expected_output,
                              const std::vector<int64_t>& mrope_section,
                              int64_t mrope_layout,
                              int64_t interleaved,
                              float scale) {
  for (const auto tensor_type : {TensorType::kFloat, TensorType::kFloat16}) {
    RunMRotaryEmbeddingTest(input_shape, input_data, position_ids, cos_cache, sin_cache, expected_output,
                            mrope_section, mrope_layout, interleaved, scale, 2, 6, tensor_type);
  }
}

const std::vector<float> kInputData = {
    1.0f,
    2.0f,
    3.0f,
    4.0f,
    5.0f,
    6.0f,
    7.0f,
    8.0f,
    9.0f,
    10.0f,
    11.0f,
    12.0f,
    13.0f,
    14.0f,
    15.0f,
    16.0f,
    17.0f,
    18.0f,
    19.0f,
    20.0f,
    21.0f,
    22.0f,
    23.0f,
    24.0f,
};

const std::vector<int64_t> kPositionIds = {
    0, 1,  // T
    1, 2,  // H
    2, 3,  // W
};

const std::vector<float> kCosCache = {
    1.00f,
    1.01f,
    1.02f,
    1.10f,
    1.11f,
    1.12f,
    1.20f,
    1.21f,
    1.22f,
    1.30f,
    1.31f,
    1.32f,
};

const std::vector<float> kSinCache = {
    0.10f,
    0.11f,
    0.12f,
    0.15f,
    0.16f,
    0.17f,
    0.20f,
    0.21f,
    0.22f,
    0.25f,
    0.26f,
    0.27f,
};

TEST(ContribOpMRotaryEmbeddingTest, SectionedRank3) {
  const std::vector<float> expected_output = {
      0.60f,
      1.42f,
      2.34f,
      4.10f,
      5.87f,
      7.98f,
      6.00f,
      7.12f,
      8.34f,
      10.70f,
      13.49f,
      16.62f,
      11.90f,
      13.37f,
      14.94f,
      19.55f,
      23.51f,
      27.81f,
      17.60f,
      19.37f,
      21.24f,
      27.05f,
      32.03f,
      37.35f,
  };

  RunMRotaryEmbeddingTests({1, 2, 12}, kInputData, kPositionIds, kCosCache, kSinCache, expected_output,
                           {1, 1, 1}, 0, 0, 1.0f);
}

TEST(ContribOpMRotaryEmbeddingTest, InterleavedRank4) {
  const std::vector<float> expected_output = {
      0.40f,
      1.05f,
      1.345f,
      2.46f,
      2.39f,
      4.21f,
      3.25f,
      4.925f,
      4.395f,
      6.995f,
      5.64f,
      9.405f,
      5.80f,
      7.65f,
      7.045f,
      10.08f,
      8.39f,
      12.85f,
      8.95f,
      12.425f,
      10.395f,
      15.515f,
      11.94f,
      18.945f,
  };

  RunMRotaryEmbeddingTests({1, 2, 2, 6}, kInputData, kPositionIds, kCosCache, kSinCache, expected_output,
                           {1, 1, 1}, 1, 1, 0.5f);
}

TEST(ContribOpMRotaryEmbeddingTest, PartialRotaryDimensionCopiesTail) {
  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> cos_cache(12, 0.0f);
  const std::vector<float> sin_cache(12, 1.0f);
  const std::vector<float> expected_output = {-4.0f, -5.0f, -6.0f, 1.0f, 2.0f, 3.0f, 7.0f, 8.0f};

  for (const auto tensor_type : {TensorType::kFloat, TensorType::kFloat16}) {
    RunMRotaryEmbeddingTest({1, 1, 8}, input_data, {0, 0, 0}, cos_cache, sin_cache, expected_output,
                            {1, 1, 1}, 0, 0, 1.0f, 1, 6, tensor_type);
  }
}

TEST(ContribOpMRotaryEmbeddingTest, RejectsOddRotaryEmbeddingDim) {
  OpTester test("MRotaryEmbedding", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(1));
  test.AddAttribute<int64_t>("rotary_embedding_dim", static_cast<int64_t>(3));
  test.AddAttribute<int64_t>("mrope_layout", static_cast<int64_t>(0));
  test.AddAttribute<int64_t>("interleaved", static_cast<int64_t>(1));
  test.AddAttribute<std::vector<int64_t>>("mrope_section", {1, 0, 0});

  test.AddInput<float>("input", {1, 1, 3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("position_ids", {3, 1, 1}, {0, 0, 0});
  test.AddInput<float>("cos_cache", {1, 1}, {1.0f});
  test.AddInput<float>("sin_cache", {1, 1}, {0.0f});
  test.AddOutput<float>("output", {1, 1, 3}, {0.0f, 0.0f, 0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "effective rotary_embedding_dim must be positive and even for non-empty inputs",
           {}, nullptr, &execution_providers);
}

TEST(ContribOpMRotaryEmbeddingTest, RejectsMropeSectionSumOverflow) {
  OpTester test("MRotaryEmbedding", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(1));
  test.AddAttribute<int64_t>("rotary_embedding_dim", static_cast<int64_t>(2));
  test.AddAttribute<int64_t>("mrope_layout", static_cast<int64_t>(1));
  test.AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
  test.AddAttribute<std::vector<int64_t>>(
      "mrope_section", {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 3});

  test.AddInput<float>("input", {1, 1, 2}, {1.0f, 2.0f});
  test.AddInput<int64_t>("position_ids", {3, 1, 1}, {0, 0, 0});
  test.AddInput<float>("cos_cache", {1, 1}, {1.0f});
  test.AddInput<float>("sin_cache", {1, 1}, {0.0f});
  test.AddOutput<float>("output", {1, 1, 2}, {0.0f, 0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "sum of 'mrope_section'",
           {}, nullptr, &execution_providers);
}

TEST(ContribOpMRotaryEmbeddingTest, RejectsNegativeRotaryEmbeddingDimAttribute) {
  auto execution_providers = GetAvailableGpuExecutionProviders();
  if (execution_providers.empty()) {
    GTEST_SKIP() << "CUDA and WebGPU execution providers are not available.";
  }

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("provider=" + ep->Type());
    auto test = CreateMRotaryEmbeddingTester(*ep);
    test->AddAttribute<int64_t>("num_heads", static_cast<int64_t>(1));
    test->AddAttribute<int64_t>("rotary_embedding_dim", static_cast<int64_t>(-1));
    test->AddAttribute<int64_t>("mrope_layout", static_cast<int64_t>(0));
    test->AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
    test->AddAttribute<std::vector<int64_t>>("mrope_section", {1, 0, 0});

    test->AddInput<float>("input", {1, 1, 2}, {1.0f, 2.0f});
    test->AddInput<int64_t>("position_ids", {3, 1, 1}, {0, 0, 0});
    test->AddInput<float>("cos_cache", {1, 1}, {1.0f});
    test->AddInput<float>("sin_cache", {1, 1}, {0.0f});
    test->AddOutput<float>("output", {1, 1, 2}, {0.0f, 0.0f});

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test->Run(OpTester::ExpectResult::kExpectFailure,
              "rotary_embedding_dim must be in range",
              {}, nullptr, &test_execution_providers);
  }
}

TEST(ContribOpMRotaryEmbeddingTest, PositionIdsOOBPassthroughAllStreams) {
  const auto available_execution_providers = GetAvailableGpuExecutionProviders();
  if (available_execution_providers.empty()) {
    GTEST_SKIP() << "CUDA and WebGPU execution providers are not available.";
  }

  const std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> cos_cache = {
      0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f};
  const std::vector<float> sin_cache = {
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f};
  const std::vector<std::vector<float>> expected_outputs = {
      {1.0f, -5.0f, -6.0f, 4.0f, 2.0f, 3.0f},
      {-4.0f, 2.0f, -6.0f, 1.0f, 5.0f, 3.0f},
      {-4.0f, -5.0f, 3.0f, 1.0f, 2.0f, 6.0f}};
  const std::vector<int64_t> invalid_position_ids = {
      std::numeric_limits<int64_t>::min(), -(int64_t{1} << 32), -1, 2, int64_t{1} << 32};

  for (int stream = 0; stream < 3; ++stream) {
    for (const int64_t invalid_position_id : invalid_position_ids) {
      std::vector<int64_t> position_ids = {0, 0, 0};
      position_ids[static_cast<size_t>(stream)] = invalid_position_id;

      for (const auto& available_ep : available_execution_providers) {
        SCOPED_TRACE("provider=" + available_ep->Type() +
                     " stream=" + std::to_string(stream) +
                     " invalid_position_id=" + std::to_string(invalid_position_id));

        auto ep = CreateGpuExecutionProvider(available_ep->Type());
        ASSERT_NE(ep, nullptr);
        auto test = CreateMRotaryEmbeddingTester(*ep);
        test->AddAttribute<int64_t>("num_heads", static_cast<int64_t>(1));
        test->AddAttribute<int64_t>("rotary_embedding_dim", static_cast<int64_t>(6));
        test->AddAttribute<int64_t>("mrope_layout", static_cast<int64_t>(0));
        test->AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
        test->AddAttribute<std::vector<int64_t>>("mrope_section", {1, 1, 1});

        test->AddInput<float>("input", {1, 1, 1, 6}, input_data);
        test->AddInput<int64_t>("position_ids", {3, 1, 1}, position_ids);
        test->AddInput<float>("cos_cache", {2, 3}, cos_cache);
        test->AddInput<float>("sin_cache", {2, 3}, sin_cache);
        test->AddOutput<float>("output", {1, 1, 1, 6}, expected_outputs[static_cast<size_t>(stream)]);

        std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
        test_execution_providers.push_back(std::move(ep));
        test->Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
      }
    }
  }
}

TEST(ContribOpMRotaryEmbeddingTest, EmptyRank3Input) {
  auto execution_providers = GetAvailableGpuExecutionProviders();
  if (execution_providers.empty()) {
    GTEST_SKIP() << "CUDA and WebGPU execution providers are not available.";
  }

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("provider=" + ep->Type());
    auto test = CreateMRotaryEmbeddingTester(*ep);
    test->AddAttribute<int64_t>("num_heads", static_cast<int64_t>(1));
    test->AddAttribute<int64_t>("mrope_layout", static_cast<int64_t>(0));
    test->AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
    test->AddAttribute<std::vector<int64_t>>("mrope_section", {0, 0, 0});

    test->AddInput<float>("input", {1, 1, 0}, std::vector<float>{});
    test->AddInput<int64_t>("position_ids", {3, 1, 1}, {0, 0, 0});
    test->AddInput<float>("cos_cache", {1, 0}, std::vector<float>{});
    test->AddInput<float>("sin_cache", {1, 0}, std::vector<float>{});
    test->AddOutput<float>("output", {1, 1, 0}, std::vector<float>{});

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test->Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
  }
}

TEST(ContribOpMRotaryEmbeddingTest, EmptyRank4Input) {
  auto execution_providers = GetAvailableGpuExecutionProviders();
  if (execution_providers.empty()) {
    GTEST_SKIP() << "CUDA and WebGPU execution providers are not available.";
  }

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("provider=" + ep->Type());
    auto test = CreateMRotaryEmbeddingTester(*ep);
    test->AddAttribute<int64_t>("num_heads", static_cast<int64_t>(2));
    test->AddAttribute<int64_t>("rotary_embedding_dim", static_cast<int64_t>(6));
    test->AddAttribute<int64_t>("mrope_layout", static_cast<int64_t>(0));
    test->AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
    test->AddAttribute<std::vector<int64_t>>("mrope_section", {1, 1, 1});

    test->AddInput<float>("input", {1, 2, 0, 6}, std::vector<float>{});
    test->AddInput<int64_t>("position_ids", {3, 1, 0}, std::vector<int64_t>{});
    test->AddInput<float>("cos_cache", {1, 3}, {1.0f, 1.0f, 1.0f});
    test->AddInput<float>("sin_cache", {1, 3}, {0.0f, 0.0f, 0.0f});
    test->AddOutput<float>("output", {1, 2, 0, 6}, std::vector<float>{});

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test->Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
