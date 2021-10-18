// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(BifurcationDetectorTest, Test1) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {1}, {2});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddInput<int64_t>("pred_tokens", {5}, {1, 5, 3, 4, 2});
  tester.AddOutput<int64_t>("tokens", {6}, {2, 1, 5, 3, 4, 2});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
