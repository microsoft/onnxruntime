// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/session/inference_session.h"
#include "core/graph/graph.h"

#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/test/training_ops/function_op_test_utils.h"

#include "test/providers/provider_test_utils.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

void OpFunctionTester::RunFunctionBodyGraphOnCPU(TwoDArray& results) {
  SetTestFunctionCalled();

  auto& model = BuildModel();
  auto& graph = model.MainGraph();
  const auto& op = Op();

  ASSERT_STATUS_OK(graph.Resolve());

  auto& node = *graph.Nodes().begin();
  ASSERT_EQ(node.OpType(), op);
  // Inline function will call Resolve itself
  ASSERT_STATUS_OK(graph.InlineFunction(node));

  // Hookup the inputs and outputs
  std::unordered_map<std::string, OrtValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  // Run the model
  SessionOptions so;
  so.session_logid = op;

  InferenceSession cpu_session_object{so, GetEnvironment()};

  // Run the inlined graph with cpu
  std::string s1;
  model.ToProto().SerializeToString(&s1);
  std::istringstream str(s1);
  ASSERT_STATUS_OK(cpu_session_object.Load(str));
  ASSERT_STATUS_OK(cpu_session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = op;
  run_options.run_log_verbosity_level = 1;

#ifdef ENABLE_TRAINING
  // Remove when training::TrainingSession class is removed.
  run_options.training_mode = true;
#endif

  std::vector<OrtValue> cpu_fetches;
  ASSERT_STATUS_OK(cpu_session_object.Run(run_options, feeds, output_names, &cpu_fetches));

  auto fetch_index = 0;

  results = TwoDArray(cpu_fetches.size());

  for (auto& fetch : cpu_fetches) {
    const Tensor& t = fetch.Get<Tensor>();
    auto float_val = t.Data<float>();
    for (auto i = 0; i < t.Shape().Size(); ++i) {
      results[fetch_index].push_back(float_val[i]);
    }
    fetch_index++;
  }
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
    test->AddAttributeProto(attr);

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
  TwoDArray results;
  ASSERT_NO_FATAL_FAILURE(inline_tester->RunFunctionBodyGraphOnCPU(results));

  // Use run_results got from inline testing as expected data,
  // test against all registered kernels.
  auto test = CreateOpTester<OpTester>(op_def,
                                       input_data, input_dims,
                                       results, output_dims,
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
