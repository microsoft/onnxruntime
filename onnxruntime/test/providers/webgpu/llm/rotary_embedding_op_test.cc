// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// Test that OOB position_ids on WebGPU pass through input unchanged (shader-side defense).
TEST(RotaryEmbeddingTest, RotaryEmbedding_PositionIds_OOB_WebGPU_Passthrough) {
  if (nullptr == DefaultWebGpuExecutionProvider().get()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  int batch_size = 1;
  int sequence_length = 1;
  int num_heads = 2;
  int head_size = 4;
  int max_sequence_length = 8;
  int hidden_size = num_heads * head_size;

  OpTester test("RotaryEmbedding", 23, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
  test.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));

  std::vector<float> input_data(hidden_size);
  for (int i = 0; i < hidden_size; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  test.AddInput<float>("input", {batch_size, sequence_length, hidden_size}, input_data);
  // Non-trivial cache values ensure pass-through (output=input) differs from valid rotary output.
  test.AddInput<float>("cos_cache", {max_sequence_length, head_size / 2},
                       std::vector<float>(max_sequence_length * head_size / 2, 0.5f));
  test.AddInput<float>("sin_cache", {max_sequence_length, head_size / 2},
                       std::vector<float>(max_sequence_length * head_size / 2, 0.866f));
  // position_id = 2048 exceeds max_sequence_length = 8 — shader passes through input unchanged.
  test.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, {2048});

  // Output should equal input when position_id is OOB (pass-through).
  test.AddOutput<float>("output", {batch_size, sequence_length, hidden_size}, input_data);
  test.SetOutputAbsErr("output", 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Test that negative position_ids pass through on WebGPU (shader-side defense catches raw_pos < 0).
TEST(RotaryEmbeddingTest, RotaryEmbedding_PositionIds_Negative_WebGPU_Passthrough) {
  if (nullptr == DefaultWebGpuExecutionProvider().get()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  int batch_size = 1;
  int sequence_length = 1;
  int num_heads = 2;
  int head_size = 4;
  int max_sequence_length = 8;
  int hidden_size = num_heads * head_size;

  OpTester test("RotaryEmbedding", 23, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
  test.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));

  std::vector<float> input_data(hidden_size);
  for (int i = 0; i < hidden_size; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  test.AddInput<float>("input", {batch_size, sequence_length, hidden_size}, input_data);
  // Non-trivial cache values ensure pass-through (output=input) differs from valid rotary output.
  test.AddInput<float>("cos_cache", {max_sequence_length, head_size / 2},
                       std::vector<float>(max_sequence_length * head_size / 2, 0.5f));
  test.AddInput<float>("sin_cache", {max_sequence_length, head_size / 2},
                       std::vector<float>(max_sequence_length * head_size / 2, 0.866f));
  // Negative position_id — shader checks raw_pos < 0 and passes through.
  test.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, {-1});

  // Output should equal input when position_id is negative (pass-through).
  test.AddOutput<float>("output", {batch_size, sequence_length, hidden_size}, input_data);
  test.SetOutputAbsErr("output", 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Test that OOB position_ids in a batch pass through on WebGPU (shader-side defense).
TEST(RotaryEmbeddingTest, RotaryEmbedding_PositionIds_OOB_InBatch_WebGPU_Passthrough) {
  if (nullptr == DefaultWebGpuExecutionProvider().get()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  int batch_size = 2;
  int sequence_length = 2;
  int num_heads = 2;
  int head_size = 4;
  int max_sequence_length = 8;
  int hidden_size = num_heads * head_size;

  OpTester test("RotaryEmbedding", 23, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("interleaved", static_cast<int64_t>(0));
  test.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));

  std::vector<float> input_data(batch_size * sequence_length * hidden_size);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  test.AddInput<float>("input", {batch_size, sequence_length, hidden_size}, input_data);
  // Non-trivial cache values ensure pass-through (output=input) differs from valid rotary output.
  test.AddInput<float>("cos_cache", {max_sequence_length, head_size / 2},
                       std::vector<float>(max_sequence_length * head_size / 2, 0.5f));
  test.AddInput<float>("sin_cache", {max_sequence_length, head_size / 2},
                       std::vector<float>(max_sequence_length * head_size / 2, 0.866f));
  // All OOB position_ids — shader passes through input unchanged.
  test.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, {100, 200, 300, 400});

  // Output should equal input when all position_ids are OOB (pass-through).
  test.AddOutput<float>("output", {batch_size, sequence_length, hidden_size}, input_data);
  test.SetOutputAbsErr("output", 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
