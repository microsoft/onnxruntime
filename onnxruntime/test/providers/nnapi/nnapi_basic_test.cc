// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
#include "core/common/logging/logging.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/test_utils.h"

#if defined(__ANDROID__)
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"
#endif

#if !defined(ORT_MINIMAL_BUILD)
// if this is a full build we need the provider test utils
#include "test/providers/provider_test_utils.h"
#endif

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

#if !defined(ORT_MINIMAL_BUILD)

// Since NNAPI EP handles Reshape and Flatten differently,
// Please see ReshapeOpBuilder::CanSkipReshape in
// <repo_root>/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/op_builder.cc
// We have a separated test for these skip reshape scenarios
TEST(NnapiExecutionProviderTest, ReshapeFlattenTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/nnapi_reshape_flatten_test.onnx");

#if defined(__ANDROID__)
  std::vector<int64_t> dims_mul_x = {2, 1, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> dims_mul_y = {3, 2, 2};
  std::vector<float> values_mul_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_y, values_mul_y,
                       &ml_value_y);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.ReshapeFlattenTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  // test load only
  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::make_unique<NnapiExecutionProvider>(0)));
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());
  ASSERT_GT(CountAssignedNodes(session_object.GetGraph(), kNnapiExecutionProvider), 0)
      << "Some nodes should have been taken by the NNAPI EP";
#endif
}

// This is to test the uint8 handling of operators without "QLinear" such as Concat and Transpose
// NNAPI will require scale and zero point for inputs of all quantized operations
// For these operators without "Qlinear", there is no information about the scale and zero point, we can
// only fetch these from the output of the previous node
// So uint8 support of these operators will only be enabled when they are internal to the graph
// by not consuming graph inputs
TEST(NnapiExecutionProviderTest, InternalUint8SupportTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/nnapi_internal_uint8_support.onnx");

#if defined(__ANDROID__)
  std::vector<int64_t> dims_x = {1, 3};
  std::vector<float> values_x = {0.0f, 256.0f, 512.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_x, values_x,
                       &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.InternalUint8SupportTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  // test load only
  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::make_unique<NnapiExecutionProvider>(0)));
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());
  ASSERT_GT(CountAssignedNodes(session_object.GetGraph(), kNnapiExecutionProvider), 0)
      << "Some nodes should have been taken by the NNAPI EP";
#endif
}

#if defined(__ANDROID__)
// This is to verify the op_builders and op_support_checkers are consistent
TEST(NnapiExecutionProviderTest, CreateOpBuilderAndOpSupportCheckerTest) {
  const auto& op_builders = nnapi::GetOpBuilders();
  const auto& op_support_checkers = nnapi::GetOpSupportCheckers();
  for (auto entry : op_builders) {
    ASSERT_TRUE(op_support_checkers.find(entry.first) != op_support_checkers.cend());
  }
  for (auto entry : op_support_checkers) {
    ASSERT_TRUE(op_builders.find(entry.first) != op_builders.cend());
  }
}
#endif  // #if defined(__ANDROID__)

TEST(NnapiExecutionProviderTest, FunctionTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("nnapi_execution_provider_test_graph.onnx");

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

#if defined(__ANDROID__)
  std::vector<int64_t> dims_mul_x = {1, 1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestNnapiExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.FunctionTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  // test load only
  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::make_unique<NnapiExecutionProvider>(0)));
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());
  ASSERT_GT(CountAssignedNodes(session_object.GetGraph(), kNnapiExecutionProvider), 0)
      << "Some nodes should have been taken by the NNAPI EP";
#endif
}

TEST(NnapiExecutionProviderTest, TestNoShapeInputModel) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("input_with_no_shape_test_graph.onnx");

  {  // Create the model with 2 add nodes, the graph has 2 inputs with no shape
    onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    // FLOAT tensor without shape
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

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

  // test load only
  // since we know NNAPI supports Add op, but both Add ops in the graph has no input shape
  // verify the entire graph will not be assigned to NNAPI EP
  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::make_unique<NnapiExecutionProvider>(0)));
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());
  ASSERT_EQ(CountAssignedNodes(session_object.GetGraph(), kNnapiExecutionProvider), 0)
      << "No node should be taken by the NNAPI EP";
}

#endif  // !(ORT_MINIMAL_BUILD

TEST(NnapiExecutionProviderTest, NNAPIFlagsTest) {
  uint32_t nnapi_flags = NNAPI_FLAG_USE_NONE;
  nnapi_flags |= NNAPI_FLAG_USE_FP16;
  onnxruntime::NnapiExecutionProvider nnapi_ep(nnapi_flags);
  const auto flags = nnapi_ep.GetNNAPIFlags();
  ASSERT_TRUE(flags & NNAPI_FLAG_USE_FP16);
  ASSERT_FALSE(flags & NNAPI_FLAG_USE_NCHW);
}

TEST(NnapiExecutionProviderTest, TestOrtFormatModel) {
  // mnist model that has only had basic optimizations applied. nnapi should be able to take at least some of the nodes
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/mnist.level1_opt.ort");

// The execution can only be performed on Android
#if defined(__ANDROID__)
  RandomValueGenerator random{};
  const std::vector<int64_t> dims = {1, 1, 28, 28};
  std::vector<float> data = random.Gaussian<float>(dims, 0.0f, 1.f);

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims, data,
                       &ml_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("Input3", ml_value));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.TestOrtFormatModel",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  // test load only
  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::make_unique<NnapiExecutionProvider>(0)));
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());
  ASSERT_GT(CountAssignedNodes(session_object.GetGraph(), kNnapiExecutionProvider), 0)
      << "Some nodes should have been taken by the NNAPI EP";
#endif
}

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
