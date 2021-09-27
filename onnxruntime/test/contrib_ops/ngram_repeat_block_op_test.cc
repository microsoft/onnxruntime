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
      {1.0f,  2.0f,  3.0f,  4.0f,
       5.0f,  6.0f,  7.0f,  8.0f,
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

TEST(BifurcationDetectorTest, Test1) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {1}, {2});
  tester.AddInput<int64_t>("find_end_idx", {}, {0});
  tester.AddInput<int64_t>("pred_tokens", {5}, {1, 5, 3, 4, 2});
  tester.AddOutput<int64_t>("tokens", {6}, {2, 1, 5, 3, 4, 2});
  tester.AddOutput<int64_t>("new_end_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
