// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(NGramRepeatBlockTest, NGramSize_3) {
  OpTester tester("NGramRepeatBlock", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("input_ids", {3, 6},
                           {0, 1, 2, 3, 1, 2,
                            0, 0, 0, 1, 0, 0,
                            0, 1, 0, 1, 0, 1});
  tester.AddInput<float>("scores", {3, 4},
                         {1.0f, 2.0f, 3.0f, 4.0f,
                          5.0f, 6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f, 12.0f});
  tester.AddAttribute("ngram_size", (int64_t)3);
  tester.AddOutput<float>("scores_out", {3, 4},
                          {1.0f, 2.0f, 3.0f, -std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 7.0f, 8.0f,
                           -std::numeric_limits<float>::infinity(), 10.0f, 11.0f, 12.0f});

  if (HasCudaEnvironment(0)) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Negative token_id used as array index causes OOB write (CPU only test).
// CUDA EP is excluded because CUDA_KERNEL_ASSERT corrupts the device context in debug builds.
TEST(NGramRepeatBlockTest, NegativeTokenId) {
  OpTester tester("NGramRepeatBlock", 1, onnxruntime::kMSDomain);

  // With ngram_size=2, the operator checks if input_ids[i] == input_ids[cur_len-1] (the tail),
  // and if so, bans token_id = input_ids[i+1]. Here input_ids[0]=1 matches input_ids[3]=1,
  // so token_id = input_ids[1] = -1000 would be used as an array index.
  tester.AddInput<int64_t>("input_ids", {1, 4}, {1, -1000, 0, 1});
  tester.AddInput<float>("scores", {1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
  tester.AddAttribute("ngram_size", (int64_t)2);
  tester.AddOutput<float>("scores_out", {1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure, "token_id", {}, nullptr, &execution_providers);
}

// Token_id >= vocab_size causes OOB write (CPU only test).
TEST(NGramRepeatBlockTest, TokenIdExceedsVocabSize) {
  OpTester tester("NGramRepeatBlock", 1, onnxruntime::kMSDomain);

  // Same logic: input_ids[0]=1 matches input_ids[3]=1, so token_id = input_ids[1] = 100.
  // vocab_size = 4 (from scores shape), so 100 >= vocab_size triggers the error.
  tester.AddInput<int64_t>("input_ids", {1, 4}, {1, 100, 0, 1});
  tester.AddInput<float>("scores", {1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
  tester.AddAttribute("ngram_size", (int64_t)2);
  tester.AddOutput<float>("scores_out", {1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure, "token_id", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
