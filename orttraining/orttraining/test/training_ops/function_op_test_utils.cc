// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/inference_session.h"
#include "core/graph/graph.h"
#include "orttraining/test/training_ops/function_op_test_utils.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace test {

TwoDArray OpFunctionTester::RunFunctionBodyGraphOnCPU() {
#ifndef NDEBUG
  run_called_ = true;
#endif

  auto p_model = BuildGraph();
  auto& graph = p_model->MainGraph();

  Status status = graph.Resolve();
  ORT_ENFORCE(status.IsOK());

  auto& node = *graph.Nodes().begin();
  ORT_ENFORCE(node.OpType() == op_);
  // Inline function will call Resolve itself
  graph.InlineFunction(node);

  // Hookup the inputs and outputs
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  // Run the model
  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  InferenceSession cpu_session_object{so, GetEnvironment()};

  // Run the inlined graph with cpu
  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::istringstream str(s1);
  status = cpu_session_object.Load(str);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = cpu_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;
  run_options.training_mode = true;

  std::vector<MLValue> cpu_fetches;
  status = cpu_session_object.Run(run_options, feeds, output_names, &cpu_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto fetch_index = 0;

  auto run_results = TwoDArray(cpu_fetches.size());

  for (auto& fetch : cpu_fetches) {
    const Tensor& t = fetch.Get<Tensor>();
    auto float_val = t.Data<float>();
    for (auto i = 0; i < t.Shape().Size(); ++i) {
      run_results[fetch_index].push_back(float_val[i]);
    }
    fetch_index++;
  }

  return run_results;
}

OpFunctionTester::~OpFunctionTester(){};

template <class T>
std::unique_ptr<T> CreateOpTester(const onnxruntime::training::OpDef& op_def,
                                  const TwoDArray& input_data,
                                  const std::vector<std::vector<int64_t>>& input_dims,
                                  const TwoDArray& expected_output_data,
                                  const std::vector<std::vector<int64_t>>& output_dims,
                                  const std::vector<AttributeProto>& attributes,
                                  int opset_version) {
  auto test = std::make_unique<T>(op_def.type.c_str(), opset_version, op_def.domain.c_str());
  for (auto attr : attributes)
    test->AddAttribute(attr.name(), attr);

  auto input_index = 0;
  for (auto& data : input_data) {
    std::string input_name = "input";
    input_name += std::to_string(input_index);
    test->template AddInput<float>(input_name.c_str(), input_dims[input_index], data);
    input_index++;
  }

  auto output_index = 0;
  for (auto& out_dims : output_dims) {
    std::string output_name = "output";
    output_name += std::to_string(output_index);
    auto& expected_data = expected_output_data[output_index];
    test->template AddOutput<float>(output_name.c_str(), out_dims, expected_data);
    output_index++;
  }
  return test;
}

void CompareResults(const onnxruntime::training::OpDef& op_def,
                    const TwoDArray& input_data,
                    const std::vector<std::vector<int64_t>>& input_dims,
                    const std::vector<std::vector<int64_t>>& output_dims,
                    const std::vector<AttributeProto>& attributes,
                    int opset_version) {
  auto inline_tester = CreateOpTester<OpFunctionTester>(op_def,
                                                        input_data, input_dims,
                                                        CreateEmpty2DArray(output_dims), output_dims,
                                                        attributes,
                                                        opset_version);
  auto run_results = inline_tester->RunFunctionBodyGraphOnCPU();

  // Use run_results got from inline testing as expected data,
  // test against all registered kernels.
  auto test = CreateOpTester<OpTester>(op_def,
                                       input_data, input_dims,
                                       run_results, output_dims,
                                       attributes,
                                       opset_version);
  test->Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

TwoDArray CreateEmpty2DArray(const std::vector<std::vector<int64_t>>& dims) {
  auto result = TwoDArray(dims.size());
  auto index = 0;
  for (auto& v : result) {
    v.resize(std::accumulate(dims[index].begin(), dims[index].end(), (int64_t)1, std::multiplies<int64_t>()));
    index++;
  }

  return result;
}

}  // namespace test
}  // namespace onnxruntime
