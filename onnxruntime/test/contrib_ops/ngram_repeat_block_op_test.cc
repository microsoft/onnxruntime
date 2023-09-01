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
#ifdef USE_ROCM
  if (nullptr != DefaultRocmExecutionProvider().get()) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultRocmExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
#endif

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
