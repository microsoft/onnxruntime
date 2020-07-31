// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "test/providers/provider_test_utils.h"
#include "gradient_checker.h"

namespace onnxruntime {
namespace test {
using TestDataVector = std::tuple<std::vector<std::vector<TensorInfo>>, // Input data
                                  std::vector<std::vector<TensorInfo>>, // output data
                                  std::vector<std::vector<onnx::AttributeProto>>>; //attribute

class GradientOpTester : public OpTester {
 public:
  explicit GradientOpTester(const char* op,
                            const std::vector<TensorInfo>& input_infos,
                            const std::vector<TensorInfo>& output_infos,
                            int opset_version = 9,
                            const char* domain = onnxruntime::kOnnxDomain,
                            bool verify_output = true)
      : OpTester(op, opset_version, domain, verify_output),
        input_infos_(input_infos),
        output_infos_(output_infos) {}

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

  std::vector<TensorInfo> input_infos_;
  std::vector<TensorInfo> output_infos_;
};
}  // namespace test
}  // namespace onnxruntime

