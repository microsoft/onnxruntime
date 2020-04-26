// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/graph/graph.h"
#include "orttraining/test/training_ops/function_op_test_utils.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace test {

namespace testdata {
std::vector<float> x = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
std::vector<float> dy(7, 1.0f);
}  // namespace testdata

TEST(FunctionOpTest, DISABLED_Gelu) {
  std::vector<std::vector<float>> input_data;
  input_data.push_back(testdata::x);

  const onnxruntime::training::OpDef& op_def{"Gelu"};
  CompareResults(op_def,
                 input_data,
                 {{7, 1}},
                 {{7, 1}},
                 {},
                 9);
}

TEST(FunctionOpTest, DISABLED_GeluGrad) {
  std::vector<std::vector<float>> input_data;
  input_data.push_back(testdata::dy);
  input_data.push_back(testdata::x);

  const onnxruntime::training::OpDef& op_def{"GeluGrad"};
  CompareResults(op_def,
                 input_data,
                 {{7, 1}, {7, 1}},
                 {{7, 1}},
                 {},
                 9);
}

}  // namespace test
}  // namespace onnxruntime
