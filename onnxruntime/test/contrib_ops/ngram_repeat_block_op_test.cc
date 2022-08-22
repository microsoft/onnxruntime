// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(NGramRepeatBlockTest, NGramSize_3_Recency_0) {
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
  tester.AddAttribute("recency_length", (int64_t)0);
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

TEST(NGramRepeatBlockTest, NGramSize_5_Recency_0) {
  OpTester tester("NGramRepeatBlock", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("input_ids", {2, 20},
                           {17, 18, 19, 20,
                            21, 22, 23, 24,
                            1, 6, 5, 4,
                            10, 9, 8, 7,
                            1, 6, 5, 4,
                            3, 2, 11, 12,
                            10, 12, 1, 13,
                            1, 14, 10, 9,
                            15, 16, 17, 7,
                            1, 6, 5, 4});
  tester.AddInput<float>("scores", {2, 12},
                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                          5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                          10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                          15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                          20.0f, 21.0f, 22.0f, 23.0f});
  tester.AddAttribute("ngram_size", (int64_t)5);
  tester.AddAttribute("recency_length", (int64_t)0);
  tester.AddOutput<float>("scores_out", {2, 12},
                          {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                           -std::numeric_limits<float>::infinity(), 11.0f, 12.0f, 13.0f, 14.0f,
                           15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                           20.0f, 21.0f, 22.0f, 23.0f});

  if (HasCudaEnvironment(0)) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(NGramRepeatBlockTest, NGramSize_5_Recency_7) {
  OpTester tester("NGramRepeatBlock", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("input_ids", {2, 20},
                           {17, 18, 19, 20,
                            21, 22, 23, 24,
                            1, 6, 5, 4,
                            10, 9, 8, 7,
                            1, 6, 5, 4,
                            3, 2, 11, 12,
                            10, 12, 1, 13,
                            1, 14, 10, 9,
                            15, 16, 17, 7,
                            1, 6, 5, 4});
  tester.AddInput<float>("scores", {2, 12},
                         {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                          5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                          10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                          15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                          20.0f, 21.0f, 22.0f, 23.0f});
  tester.AddAttribute("ngram_size", (int64_t)5);
  tester.AddAttribute("recency_length", (int64_t)7);
  tester.AddOutput<float>("scores_out", {2, 12},
                          {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                           10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                           15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                           20.0f, 21.0f, 22.0f, 23.0f});

  if (HasCudaEnvironment(0)) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
