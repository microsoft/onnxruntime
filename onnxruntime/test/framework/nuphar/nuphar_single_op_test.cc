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

TEST(NupharTest, PadTest) {
#ifdef USE_TVM_WITH_LLVM

  // This is a test with Pad->Add
  // Create Model
  onnxruntime::Model model("nuphar_only_pad_test5");
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
  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &type_proto_shorter);
  inputs.push_back(&input_arg_1);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("Out", &type_proto);
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

  string model_file_name = "nuphar_only_pad_test5.onnx";
  EXPECT_TRUE(graph.Resolve().IsOK());
  EXPECT_TRUE(onnxruntime::Model::Save(model, model_file_name).IsOK());

  // Prepare Input
  vector<int64_t> dim_in = {len - 1};
  vector<float> val_in((size_t)len - 1, 1.0f);

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
  auto ret = session.Run(feeds, output_names, &result);
  EXPECT_TRUE(ret.IsOK());

  // Verfiy outputs
  vector<int64_t> dim_out = {len};
  vector<float> val_out((size_t)len, 1.0f);
  val_out[len - 1] = 0.0f;

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
