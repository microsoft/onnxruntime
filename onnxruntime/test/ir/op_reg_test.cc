// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {
TEST(FeatureVectorizerTest, TraditionalMlOpTest) {
  Model model("traditionalMl");
  auto& graph = model.MainGraph();

  // Case: A traditional ml graph.
  //                           SouceNode
  //                              |
  //                       node_1(CastMap)
  //                      (tensor(float))
  //                             |
  //                    node_5 (FeatureVectorizer)
  //                              |
  //                           SinkNode

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // Type: tensor(float)
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  // Type: map(int64,float);
  TypeProto map_int64_float;
  auto map_type = map_int64_float.mutable_map_type();
  map_type->set_key_type(TensorProto::INT64);
  auto map_value_type = map_type->mutable_value_type()->mutable_tensor_type();
  map_value_type->set_elem_type(TensorProto::FLOAT);
  map_value_type->mutable_shape();

  NodeArg* input_arg1 = new NodeArg("node_1_in_1", &map_int64_float);
  inputs.clear();
  inputs.push_back(input_arg1);
  NodeArg* output_arg1 = new NodeArg("node_1_out_1", &tensor_float);
  outputs.clear();
  outputs.push_back(output_arg1);
  graph.AddNode("node_1", "CastMap", "node 1", inputs, outputs, nullptr, kMLDomain);

  inputs.clear();
  inputs.push_back(output_arg1);

  NodeArg* output_arg4 = new NodeArg("node_4_out_1", &tensor_float);
  outputs.clear();
  outputs.push_back(output_arg4);
  graph.AddNode("node_4", "FeatureVectorizer", "node 4", inputs, outputs, nullptr, kMLDomain);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  delete input_arg1;
  delete output_arg1;
  delete output_arg4;
}
}  // namespace test
}  // namespace onnxruntime
