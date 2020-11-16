// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/inference_session_wrapper.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

#ifdef __ANDROID__
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<float>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.template Data<float>(), rtensor.template Data<float>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}
#endif

void RunAndVerifyOutputs(const std::string& model_file_name,
                         const char* log_id,
                         const NameMLValMap& feeds,
                         const std::vector<std::string>& output_names,
                         const std::vector<int64_t>& expected_dims,
                         const std::vector<float>& expected_values) {
  SessionOptions so;
  so.session_logid = log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(onnxruntime::make_unique<::onnxruntime::NnapiExecutionProvider>(0)));
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Since we already know the model is entirely supported by NNAPI, all nodes (2 Add nodes here) will be fused
  // Get the graph after session is initialized, and verify the fused node (the only node in the graph) is using NNAPI EP
  const auto& graph = session_object.GetGraph();
  ASSERT_EQ(1, graph.NumberOfNodes());  // Make sure the graph has 1 fused node
  ASSERT_EQ(onnxruntime::kNnapiExecutionProvider, graph.Nodes().cbegin()->GetExecutionProviderType());

// The execution can only be performed on Android
#ifdef __ANDROID__
  // Now run and verify the result
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));
  VerifyOutputs(fetches, expected_dims, expected_values);
#else
  ORT_UNUSED_PARAMETER(feeds);
  ORT_UNUSED_PARAMETER(output_names);
  ORT_UNUSED_PARAMETER(expected_dims);
  ORT_UNUSED_PARAMETER(expected_values);
#endif
}

// Since NNAPI EP handles Reshape and Flatten differently,
// Please see ReshapeOpBuilder::CanSkipReshape in <repo_root>/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/op_builder.cc
// We have a separated test for these skip reshape scenarios
TEST(NnapiExecutionProviderTest, ReshapeFlattenTest) {
  std::string model_file_name = "testdata/nnapi_reshape_flatten_test.onnx";

  std::vector<int64_t> dims_mul_x = {2, 1, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> dims_mul_y = {3, 2, 2};
  std::vector<float> values_mul_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_y, values_mul_y, &ml_value_y);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Z");

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_z = {1, 6};
  std::vector<float> expected_values_mul_z = {59.0f, 72.0f, 129.0f, 159.0f, 204.0f, 253.0f};

  RunAndVerifyOutputs(model_file_name, "NnapiExecutionProviderTest.ReshapeFlattenTest", feeds, output_names, expected_dims_mul_z, expected_values_mul_z);
}

TEST(NnapiExecutionProviderTest, FunctionTest) {
  std::string model_file_name = "nnapi_execution_provider_test_graph.onnx";
  {  // Create the model with 2 add nodes
    onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    // FLOAT tensor.
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
    inputs.push_back(&input_arg_1);
    inputs.push_back(&input_arg_2);
    auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
    outputs.push_back(&output_arg);
    graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

    auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
    inputs.clear();
    inputs.push_back(&output_arg);
    inputs.push_back(&input_arg_3);
    auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_2);
    graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_file_name));
  }

  std::vector<int64_t> dims_mul_x = {1, 1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 1, 3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  RunAndVerifyOutputs(model_file_name, "NnapiExecutionProviderTest.FunctionTest", feeds, output_names, expected_dims_mul_m, expected_values_mul_m);
}

TEST(NnapiExecutionProviderTest, NNAPIFlagsTest) {
  unsigned long nnapi_flags = NNAPI_FLAG_USE_NONE;
  nnapi_flags |= NNAPI_FLAG_USE_FP16;
  onnxruntime::NnapiExecutionProvider nnapi_ep(nnapi_flags);
  const auto flags = nnapi_ep.GetNNAPIFlags();
  ASSERT_TRUE(flags & NNAPI_FLAG_USE_FP16);
  ASSERT_FALSE(flags & NNAPI_FLAG_USE_NCHW);
}

}  // namespace test
}  // namespace onnxruntime
