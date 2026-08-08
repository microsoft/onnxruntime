// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

  for (auto& ep : execution_providers) {
    OpTester test("MRotaryEmbedding", 1, kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    test.AddAttribute<int64_t>("rotary_embedding_dim", rotary_embedding_dim);
    test.AddAttribute<int64_t>("mrope_layout", mrope_layout);
    test.AddAttribute<int64_t>("interleaved", interleaved);
    test.AddAttribute<float>("scale", scale);
    test.AddAttribute<std::vector<int64_t>>("mrope_section", mrope_section);

    if (tensor_type == TensorType::kFloat) {
      test.AddInput<float>("input", input_shape, input_data);
      test.AddInput<int64_t>("position_ids", position_ids_shape, position_ids);
      test.AddInput<float>("cos_cache", cache_shape, cos_cache);
      test.AddInput<float>("sin_cache", cache_shape, sin_cache);
      test.AddOutput<float>("output", input_shape, expected_output);
    } else {
      test.AddInput<MLFloat16>("input", input_shape, ToFloat16(input_data));
      test.AddInput<int64_t>("position_ids", position_ids_shape, position_ids);
      test.AddInput<MLFloat16>("cos_cache", cache_shape, ToFloat16(cos_cache));
      test.AddInput<MLFloat16>("sin_cache", cache_shape, ToFloat16(sin_cache));
      test.AddOutput<MLFloat16>("output", input_shape, ToFloat16(expected_output));
      test.SetOutputAbsErr("output", 0.01f);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
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

}  // namespace
}  // namespace test
}  // namespace onnxruntime
