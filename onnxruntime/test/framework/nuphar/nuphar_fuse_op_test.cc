// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/capturing_sink.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/default_providers.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {
// Nuphar Fused Op Test
TEST(NupharTest, FuseAddTest) {
#ifdef USE_TVM_WITH_LLVM
  // Create Model
  onnxruntime::Model model("nuphar_fused_add_test");
  onnxruntime::Graph& graph = model.MainGraph();

  TypeProto type_proto_scalar;
  type_proto_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);
  type_proto_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);  // dim {1, }

  TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);
  type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(6);  // dim {6, }

  vector<onnxruntime::NodeArg*> inputs;
  vector<onnxruntime::NodeArg*> outputs;

  // Construct Node 1
  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &type_proto);
  inputs.push_back(&input_arg_1);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &type_proto_scalar);
  inputs.push_back(&input_arg_2);
  TensorProto init_Y;
  init_Y.set_name("Y");
  init_Y.set_data_type(TensorProto_DataType_DOUBLE);
  init_Y.add_double_data(10.0);
  init_Y.add_dims(1);
  graph.AddInitializedTensor(init_Y);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("node1_out", &type_proto);
  outputs.push_back(&output_arg_1);

  graph.AddNode("node1", "Add", "Add operator", inputs, outputs);
  inputs.clear();
  outputs.clear();

  // Construct Node 2
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &type_proto_scalar);
  TensorProto init_Z;
  init_Z.set_name("Z");
  init_Z.set_data_type(TensorProto_DataType_DOUBLE);
  init_Z.add_double_data(100.0);
  init_Z.add_dims(1);
  graph.AddInitializedTensor(init_Z);
  inputs.push_back(&input_arg_3);
  inputs.push_back(&output_arg_1);  //another input from node1 output

  auto& output_arg_2 = graph.GetOrCreateNodeArg("Out", &type_proto);
  outputs.push_back(&output_arg_2);

  graph.AddNode("node2", "Add", "Add operator", inputs, outputs);
  inputs.clear();
  outputs.clear();

  string model_file_name = "nuphar_fused_add_test.onnx";
  EXPECT_TRUE(graph.Resolve().IsOK());
  EXPECT_TRUE(onnxruntime::Model::Save(model, model_file_name).IsOK());

  // Prepare Input
  vector<int64_t> dim_in = {6};
  vector<double> val_in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  auto provider = DefaultNupharExecutionProvider();
  auto allocator = provider->GetAllocator(0, OrtMemTypeDefault);
  MLValue ml_value_x;
  CreateMLValue<double>(allocator, dim_in, val_in, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(make_pair("X", ml_value_x));

  // Prepare output
  vector<string> output_names;
  output_names.push_back("Out");
  vector<MLValue> result;

  // Create Session
  SessionOptions so;
  InferenceSession session(so);

  EXPECT_TRUE(session.RegisterExecutionProvider(std::move(provider)).IsOK());
  EXPECT_TRUE(session.Load(model_file_name).IsOK());
  EXPECT_TRUE(session.Initialize().IsOK());
  // Run session
  EXPECT_TRUE(session.Run(feeds, output_names, &result).IsOK());

  // Verfiy outputs
  vector<int64_t> dim_out = {6};
  vector<double> val_out = {111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f};

  EXPECT_TRUE(1 == result.size());
  const Tensor& resTensor = result.front().Get<Tensor>();
  TensorShape resShape(dim_out);
  EXPECT_EQ(resTensor.Shape(), resShape);
  EXPECT_EQ(resTensor.DataType(), DataTypeImpl::GetType<double>());

  for (size_t i = 0; i < val_out.size(); ++i) {
    EXPECT_EQ(resTensor.Data<double>()[i], val_out[i]);
  }
#endif
}

TEST(NupharTest, FuseAddWithBroadcastTest) {
#ifdef USE_TVM_WITH_LLVM
  // Create Model
  onnxruntime::Model model("nuphar_fused_add_with_broadcast_test");
  onnxruntime::Graph& graph = model.MainGraph();

  TypeProto type_proto_scalar;
  type_proto_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type_proto_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);  // dim {1, }

  int64_t len = 32;
  int64_t batch = 2;

  TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch);
  type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(len);  // dim {2, 32}

  TypeProto type_proto_shorter;
  type_proto_shorter.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type_proto_shorter.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch);
  type_proto_shorter.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);  // dim {2, 1}

  vector<onnxruntime::NodeArg*> inputs;
  vector<onnxruntime::NodeArg*> outputs;

  // Construct Node 1
  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &type_proto);
  inputs.push_back(&input_arg_1);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &type_proto_shorter);
  inputs.push_back(&input_arg_2);
  TensorProto init_Y;
  init_Y.set_name("Y");
  init_Y.set_data_type(TensorProto_DataType_FLOAT);
  for (int i = 0; i < batch; i++)
    init_Y.add_float_data(10.0f);
  init_Y.add_dims(batch);
  init_Y.add_dims(1);
  graph.AddInitializedTensor(init_Y);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("node1_out", &type_proto);
  outputs.push_back(&output_arg_1);

  graph.AddNode("node1", "Add", "Add operator", inputs, outputs);
  inputs.clear();
  outputs.clear();

  // Construct Node 2
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &type_proto_scalar);
  TensorProto init_Z;
  init_Z.set_name("Z");
  init_Z.set_data_type(TensorProto_DataType_FLOAT);
  init_Z.add_float_data(100.0f);
  init_Z.add_dims(1);
  graph.AddInitializedTensor(init_Z);
  inputs.push_back(&input_arg_3);
  inputs.push_back(&output_arg_1);  //another input from node1 output

  auto& output_arg_2 = graph.GetOrCreateNodeArg("Out", &type_proto);
  outputs.push_back(&output_arg_2);

  graph.AddNode("node2", "Add", "Add operator", inputs, outputs);
  inputs.clear();
  outputs.clear();

  string model_file_name = "nuphar_fused_add_with_broadcast_test2.onnx";
  EXPECT_TRUE(graph.Resolve().IsOK());
  EXPECT_TRUE(onnxruntime::Model::Save(model, model_file_name).IsOK());

  // Prepare Input
  vector<int64_t> dim_in = {batch, len};
  vector<float> val_in((size_t)len * batch, 1.0f);

  auto provider = DefaultNupharExecutionProvider();
  auto allocator = provider->GetAllocator(0, OrtMemTypeDefault);
  MLValue ml_value_x;
  CreateMLValue<float>(allocator, dim_in, val_in, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(make_pair("X", ml_value_x));

  // Prepare output
  vector<string> output_names;
  output_names.push_back("Out");
  vector<MLValue> result;

  // Create Session
  SessionOptions so;
  InferenceSession session(so);

  EXPECT_TRUE(session.RegisterExecutionProvider(std::move(provider)).IsOK());
  EXPECT_TRUE(session.Load(model_file_name).IsOK());
  EXPECT_TRUE(session.Initialize().IsOK());

  // Run session
  EXPECT_TRUE(session.Run(feeds, output_names, &result).IsOK());

  // Verfiy outputs
  vector<int64_t> dim_out = {batch, len};
  vector<float> val_out((size_t)len * batch, 111.0f);

  EXPECT_TRUE(1 == result.size());
  const Tensor& resTensor = result.front().Get<Tensor>();
  TensorShape resShape(dim_out);
  EXPECT_EQ(resTensor.Shape(), resShape);
  EXPECT_EQ(resTensor.DataType(), DataTypeImpl::GetType<float>());

  for (size_t i = 0; i < val_out.size(); ++i) {
    EXPECT_EQ(resTensor.Data<float>()[i], val_out[i]);
  }
#endif
}

TEST(NupharTest, FuseAddAndPadTest) {
#ifdef USE_TVM_WITH_LLVM

  // This is a test with Pad->Add
  // Create Model
  onnxruntime::Model model("nuphar_fused_add_with_pad_test");
  onnxruntime::Graph& graph = model.MainGraph();

  int64_t len = 32;

  TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(len);  // dim {32}

  TypeProto type_proto_shorter;
  type_proto_shorter.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type_proto_shorter.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(len - 1);  // dim {31}

  vector<onnxruntime::NodeArg*> inputs;
  vector<onnxruntime::NodeArg*> outputs;

  // Construct Node 1 (Pad)
  auto& input_arg_pad_1 = graph.GetOrCreateNodeArg("Y", &type_proto_shorter);
  inputs.push_back(&input_arg_pad_1);
  TensorProto init_Y;
  init_Y.set_name("Y");
  init_Y.set_data_type(TensorProto_DataType_FLOAT);
  for (int i = 0; i < len - 1; i++)
    init_Y.add_float_data(0);
  init_Y.add_dims(len - 1);
  graph.AddInitializedTensor(init_Y);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("node_pad_out", &type_proto);
  outputs.push_back(&output_arg_1);
  AttributeProto attr_pad;
  attr_pad.set_name("pads");
  attr_pad.set_type(AttributeProto_AttributeType_INTS);
  attr_pad.add_ints(0);
  attr_pad.add_ints(1);

  std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto> attributes;
  attributes["pads"] = attr_pad;
  graph.AddNode("node1", "Pad", "Pad operator", inputs, outputs, &attributes, "");
  inputs.clear();
  outputs.clear();

  // Construct Node 2 (Add)
  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &type_proto);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&output_arg_1);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("Out", &type_proto);
  outputs.push_back(&output_arg_2);
  graph.AddNode("node2", "Add", "Add operator", inputs, outputs);
  inputs.clear();
  outputs.clear();

  string model_file_name = "nuphar_fused_add_with_pad_test.onnx";
  EXPECT_TRUE(graph.Resolve().IsOK());
  EXPECT_TRUE(onnxruntime::Model::Save(model, model_file_name).IsOK());

  // Prepare Input
  vector<int64_t> dim_in = {len};
  vector<float> val_in((size_t)len, 1.0f);

  auto provider = DefaultNupharExecutionProvider();
  auto allocator = provider->GetAllocator(0, OrtMemTypeDefault);
  MLValue ml_value_x;
  CreateMLValue<float>(allocator, dim_in, val_in, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(make_pair("X", ml_value_x));

  // Prepare output
  vector<string> output_names;
  output_names.push_back("Out");
  vector<MLValue> result;

  // Create Session
  SessionOptions so;
  InferenceSession session(so);

  EXPECT_TRUE(session.RegisterExecutionProvider(std::move(provider)).IsOK());
  EXPECT_TRUE(session.Load(model_file_name).IsOK());
  EXPECT_TRUE(session.Initialize().IsOK());

  // Run session
  EXPECT_TRUE(session.Run(feeds, output_names, &result).IsOK());

  // Verfiy outputs
  vector<int64_t> dim_out = {len};
  vector<float> val_out((size_t)len, 1.0f);

  EXPECT_TRUE(1 == result.size());
  const Tensor& resTensor = result.front().Get<Tensor>();
  TensorShape resShape(dim_out);
  EXPECT_EQ(resTensor.Shape(), resShape);
  EXPECT_EQ(resTensor.DataType(), DataTypeImpl::GetType<float>());

  for (size_t i = 0; i < val_out.size(); ++i) {
    EXPECT_EQ(resTensor.Data<float>()[i], val_out[i]);
  }

#endif
}
}  // namespace test
}  // namespace onnxruntime
