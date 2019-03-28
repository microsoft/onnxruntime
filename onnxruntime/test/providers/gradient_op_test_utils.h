// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "provider_test_utils.h"

namespace onnxruntime {
namespace test {

class GradientOpTester : public OpTester {
 public:
  using OpTester::OpTester;

  void Run(int output_index_to_use_as_loss,
           int data_index_of_output,
           ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr);

 private:
  void FillFeedsAndOutputNames(std::unordered_map<std::string, MLValue>& feeds,
                               std::vector<std::string>& output_names,
                               int output_index_to_use_as_loss,
                               int data_index_of_output);
};
}  // namespace test
}  // namespace onnxruntime
