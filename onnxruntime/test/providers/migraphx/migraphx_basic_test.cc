// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/providers/migraphx/migraphx_execution_provider_utils.h"
#include <string>
#include <thread>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with two inputs and one initializer.
 * input: "X", "Y" and "Z"
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Ini"  Add
 *     | \  /
 *     |  Add
 *     |      \
 *      \     Shape
 *       \    /
 *       Reshape
 *          |
 *          M
 */
void CreateBaseModel(onnxruntime::Model& model, std::vector<int> dims) {
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  for (auto dim : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  // INT tensor
  ONNX_NAMESPACE::TypeProto int64_tensor;
  int64_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  // constant
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  value_tensor.set_name("Ini");
  graph.AddInitializedTensor(value_tensor);

  // Create node1 (Add)
  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  // Create node2 (Add)
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Ini", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  // Create node3 (Shape)
  inputs.clear();
  outputs.clear();
  inputs.push_back(&output_arg_2);
  auto& output_arg_3 = graph.GetOrCreateNodeArg("S", &int64_tensor);
  outputs.push_back(&output_arg_3);
  graph.AddNode("node_3", "Shape", "node 3.", inputs, outputs);

  // Create node4 (Reshape)
  inputs.clear();
  outputs.clear();
  inputs.push_back(&input_arg_3);
  inputs.push_back(&output_arg_3);
  auto& output_arg_4 = graph.GetOrCreateNodeArg("R", &float_tensor);
  outputs.push_back(&output_arg_4);
  graph.AddNode("node_4", "Reshape", "node 4.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
}

TEST(MIGraphXExecutionProviderTest, GraphInputName) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  ASSERT_EQ(IsGraphInput(gv, "X"), true);
}

TEST(MIGraphXExecutionProviderTest, GraphInitializer) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  ASSERT_EQ(IsGraphInitializer(gv, "Ini"), true);
}

TEST(MIGraphXExecutionProviderTest, NodeInputNum) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node0 = gv.GetNode(0);
  const auto& node1 = gv.GetNode(1);

  ASSERT_EQ(getNodeInputNum(*node0), 0);
  ASSERT_EQ(getNodeInputNum(*node1), 1);
}

TEST(MIGraphXExecutionProviderTest, IsNodeInput) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node2 = gv.GetNode(1);
  ASSERT_EQ(isInputNode(node2, "M"), true);
}

TEST(MIGraphXExecutionProviderTest, canEvalArgument) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node2 = gv.GetNode(3);
  std::vector<NodeIndex> input_nodes;
  ASSERT_EQ(canEvalNodeArgument(gv, node2, {1}, input_nodes), true);
}

}  // namespace test
}  // namespace onnxruntime
