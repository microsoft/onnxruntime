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

TEST(BifurcationDetectorTest, Test2) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("src_tokens", {26}, {756,  194,   39, 1015, 5529, 1216,   24,   72,   23, 1976, 6174, 1340,
           6,   39,  194, 2161, 1480, 4955,    8, 7806,   65, 1091,    8,  560,
        4077,  196});
  tester.AddInput<int64_t>("cur_tokens", {6}, {2,  756,  194,   39, 8155,   23});
  tester.AddInput<int64_t>("find_end_idx", {}, {0});
  tester.AddOutput<int64_t>("tokens", {6}, {2,  756,  194,   39, 8155,   23});
  tester.AddOutput<int64_t>("new_end_idx", {}, {9});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
