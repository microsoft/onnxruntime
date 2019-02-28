// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {
namespace test {

typedef std::vector<onnxruntime::NodeArg*> ArgMap;

size_t CountCopyNodes(const onnxruntime::Graph& graph) {
  size_t num_copy_nodes = 0;
  for (auto& p : graph.Nodes())
    num_copy_nodes += (p.OpType().substr(0, 6) == "Memcpy");
  return num_copy_nodes;
}

static common::Status LoadInferenceSessionFromModel(InferenceSession& session, onnxruntime::Model& model) {
  std::stringstream s1;
  model.ToProto().SerializeToOstream(&s1);
  return session.Load(s1);
}

#define CREATE_INITIALIZER_FUNC(T, PROTO_DATATYPE, PROTO_ADD_DATA)                                          \
  onnxruntime::NodeArg& CreateInitializer(onnxruntime::Graph& graph, const std::string& name,               \
                                          const std::vector<int64_t>& shape, const std::vector<T>& value) { \
    ONNX_NAMESPACE::TensorProto tensor_proto;                                                               \
    for (auto dim : shape) tensor_proto.add_dims(dim);                                                      \
    tensor_proto.set_data_type(PROTO_DATATYPE);                                                             \
    for (auto v : value) tensor_proto.PROTO_ADD_DATA(v);                                                    \
    tensor_proto.set_name(name);                                                                            \
    graph.AddInitializedTensor(tensor_proto);                                                               \
    TypeProto type_proto;                                                                                   \
    type_proto.mutable_tensor_type()->set_elem_type(PROTO_DATATYPE);                                        \
    return graph.GetOrCreateNodeArg(name, &type_proto);                                                     \
  }

CREATE_INITIALIZER_FUNC(float, TensorProto_DataType_FLOAT, add_float_data)
CREATE_INITIALIZER_FUNC(int64_t, TensorProto_DataType_INT64, add_int64_data)
// TO DO: Figure out a way to enable it again
TEST(CUDAFenceTests, DISABLED_PartOnCPU) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test");
  onnxruntime::Graph& graph = model->MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg x1_def("X1", &tensor_float);
  onnxruntime::NodeArg y_def("Y", &tensor_float);
  onnxruntime::NodeArg z_def("Z", &tensor_float);
  onnxruntime::NodeArg out_def("Out", &tensor_float);

  auto& w_def = CreateInitializer(graph, "W", std::vector<int64_t>({2, 2}), std::vector<float>({-1, 2, 3, -4}));

  graph.AddNode("node1", "MatMul", "MatMul operator", ArgMap{&w_def, &x1_def}, ArgMap{&y_def})
      .SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);
  graph.AddNode("node2", "Add", "Add operator", ArgMap{&y_def, &w_def}, ArgMap{&z_def})
      .SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  graph.AddNode("node3", "Add", "Add operator", ArgMap{&y_def, &z_def}, ArgMap{&out_def})
      .SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);

  // add and then delete a node to test node iteration against nullptr
  auto& node = graph.AddNode("node_to_delete", "Add", "Add operator", ArgMap{&y_def, &z_def}, ArgMap{&out_def});
  graph.RemoveNode(node.Index());

  ASSERT_TRUE(graph.Resolve().IsOK());

  auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  auto element_type = DataTypeImpl::GetType<float>();
  TensorShape shape({2, 2});
  float data[4] = {-1, 2, 3, -4};

  //create fake ml value with owned buffer.
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      element_type,
      shape,
      cpu_allocator);
  memcpy(p_tensor->MutableData<float>(), data, sizeof(data));
  MLValue value;
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;
  InferenceSession session(so);
  LoadInferenceSessionFromModel(session, *model);
  CUDAExecutionProviderInfo xp_info;
  session.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info));
  ASSERT_TRUE(session.Initialize().IsOK());
  ASSERT_TRUE(1 == CountCopyNodes(graph));

  vector<MLValue> outputs;
  session.Run(std::unordered_map<std::string, MLValue>{{"X1", value}},
              std::vector<std::string>{"Out"},
              &outputs);
  ASSERT_TRUE(1 == outputs.size());
  const Tensor& output = outputs[0].Get<Tensor>();
  EXPECT_EQ(output.Shape(), shape);
  EXPECT_EQ(output.DataType(), DataTypeImpl::GetType<float>());

  float expected_output[4] = {13.0f, -18.0f, -27.0f, 40.0f};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(output.template Data<float>()[i], expected_output[i]);
  }
}

TEST(CUDAFenceTests, TileWithInitializer) {
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test");
  onnxruntime::Graph& graph = model->MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg x1_def("X1", &tensor_float);
  onnxruntime::NodeArg y_def("Y", &tensor_float);
  auto& tile_repeat_def = CreateInitializer(graph, "tile_repeat", std::vector<int64_t>({2}), std::vector<int64_t>({1, 2}));

  graph.AddNode("node1", "Tile", "Tile operator", ArgMap{&x1_def, &tile_repeat_def}, ArgMap{&y_def})
      .SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);

  ASSERT_TRUE(graph.Resolve().IsOK());
  ASSERT_TRUE(0 == CountCopyNodes(graph));

  auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  auto element_type = DataTypeImpl::GetType<float>();
  TensorShape shape({2, 2});
  float data[4] = {-1, 2, 3, -4};

  //create fake ml value with owned buffer.
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      element_type,
      shape,
      cpu_allocator);
  memcpy(p_tensor->MutableData<float>(), data, sizeof(data));

  MLValue value;
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;
  InferenceSession session(so);
  LoadInferenceSessionFromModel(session, *model);
  CUDAExecutionProviderInfo xp_info;
  session.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info));
  ASSERT_TRUE(session.Initialize().IsOK());

  vector<MLValue> outputs;
  session.Run(std::unordered_map<std::string, MLValue>{{"X1", value}},
              std::vector<std::string>{"Y"},
              &outputs);
  ASSERT_TRUE(1 == outputs.size());
  const Tensor& output = outputs[0].Get<Tensor>();
  EXPECT_EQ(output.Shape(), TensorShape({2, 4}));
  EXPECT_EQ(output.DataType(), DataTypeImpl::GetType<float>());

  float expected_output[8] = {-1, 2, -1, 2, 3, -4, 3, -4};
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(output.template Data<float>()[i], expected_output[i]);
  }
}

TEST(CUDAFenceTests, TileWithComputedInput) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  std::unique_ptr<onnxruntime::Model> model = std::make_unique<onnxruntime::Model>("test", true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version);
  onnxruntime::Graph& graph = model->MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_int64;
  tensor_int64.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  onnxruntime::NodeArg x1_def("X1", &tensor_float);
  onnxruntime::NodeArg y_def("Y", &tensor_float);
  onnxruntime::NodeArg s_def("S", &tensor_int64);
  onnxruntime::NodeArg out_def("Out", &tensor_float);
  auto& w_def = CreateInitializer(graph, "W", std::vector<int64_t>({2, 2}), std::vector<float>({-1, 2, 3, -4}));

  graph.AddNode("node1", "MatMul", "MatMul operator", ArgMap{&x1_def, &w_def}, ArgMap{&y_def})
      .SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);
  graph.AddNode("node2", "Shape", "Shape operator", ArgMap{&y_def}, ArgMap{&s_def})
      .SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  graph.AddNode("node3", "Tile", "Tile operator", ArgMap{&y_def, &s_def}, ArgMap{&out_def})
      .SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);

  ASSERT_TRUE(graph.Resolve().IsOK());
  ASSERT_TRUE(0 == CountCopyNodes(graph));

  auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  auto element_type = DataTypeImpl::GetType<float>();
  TensorShape shape({2, 2});
  float data[4] = {-1, 2, 3, -4};

  //create fake ml value with owned buffer.
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      element_type,
      shape,
      cpu_allocator);
  memcpy(p_tensor->MutableData<float>(), data, sizeof(data));

  MLValue value;
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;
  InferenceSession session(so);
  LoadInferenceSessionFromModel(session, *model);
  CUDAExecutionProviderInfo xp_info;
  session.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info));
  ASSERT_TRUE(session.Initialize().IsOK());

  vector<MLValue> outputs;
  session.Run(std::unordered_map<std::string, MLValue>{{"X1", value}},
              std::vector<std::string>{"Out"},
              &outputs);
  ASSERT_TRUE(1 == outputs.size());
  const Tensor& output = outputs[0].Get<Tensor>();
  EXPECT_EQ(output.Shape(), TensorShape({4, 4}));
  EXPECT_EQ(output.DataType(), DataTypeImpl::GetType<float>());

  float expected_output[16] = {7, -10, 7, -10, -15, 22, -15, 22, 7, -10, 7, -10, -15, 22, -15, 22};
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(output.template Data<float>()[i], expected_output[i]);
  }
}

}  // namespace test
}  // namespace onnxruntime
