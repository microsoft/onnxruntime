// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"

#include "test/util/include/current_test_name.h"
#include "test/util/include/test_utils.h"

#include "test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

class TelumEndToEndTest : public TelumTestBase {};

namespace {

using ModelProto = ONNX_NAMESPACE::ModelProto;
using GraphProto = ONNX_NAMESPACE::GraphProto;

constexpr int64_t kBatch = 2;
constexpr int64_t kIn = 3;
constexpr int64_t kHidden = 4;

void AddTensorTypeAndShape(ONNX_NAMESPACE::TypeProto& type_proto,
                           int32_t elem_type,
                           const std::vector<int64_t>& dims) {
  auto* tensor_type = type_proto.mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);
  auto* shape = tensor_type->mutable_shape();
  shape->clear_dim();
  for (int64_t d : dims) {
    shape->add_dim()->set_dim_value(d);
  }
}

void AddGraphInput(GraphProto& graph,
                   const std::string& name,
                   const std::vector<int64_t>& dims) {
  auto* vi = graph.add_input();
  vi->set_name(name);
  AddTensorTypeAndShape(*vi->mutable_type(),
                        ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                        dims);
}

void AddGraphOutput(GraphProto& graph,
                    const std::string& name,
                    const std::vector<int64_t>& dims) {
  auto* vi = graph.add_output();
  vi->set_name(name);
  AddTensorTypeAndShape(*vi->mutable_type(),
                        ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                        dims);
}

void AddValueInfo(GraphProto& graph,
                  const std::string& name,
                  const std::vector<int64_t>& dims) {
  auto* vi = graph.add_value_info();
  vi->set_name(name);
  AddTensorTypeAndShape(*vi->mutable_type(),
                        ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                        dims);
}

void AddFloatInitializer(GraphProto& graph,
                         const std::string& name,
                         const std::vector<int64_t>& dims,
                         const std::vector<float>& values) {
  int64_t expected = 1;
  for (int64_t d : dims) expected *= d;
  ORT_ENFORCE(expected == static_cast<int64_t>(values.size()));

  auto* init = graph.add_initializer();
  init->set_name(name);
  init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t d : dims) init->add_dims(d);
  for (float v : values) init->add_float_data(v);
}

void AddIntAttr(ONNX_NAMESPACE::NodeProto& node, const std::string& name, int64_t v) {
  auto* a = node.add_attribute();
  a->set_name(name);
  a->set_type(ONNX_NAMESPACE::AttributeProto::INT);
  a->set_i(v);
}

void AddFloatAttr(ONNX_NAMESPACE::NodeProto& node, const std::string& name, float v) {
  auto* a = node.add_attribute();
  a->set_name(name);
  a->set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
  a->set_f(v);
}

ModelProto CreateMiniTransformerBlockModel() {
  ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.set_producer_name("onnxruntime-telum-tests");
  model.set_producer_version("0");

  // Opset imports. LayerNormalization is in ONNX opset 17.
  {
    auto* onnx_opset = model.add_opset_import();
    onnx_opset->set_domain("");  // ONNX domain
    onnx_opset->set_version(17);

    auto* ms_opset = model.add_opset_import();
    ms_opset->set_domain(onnxruntime::kMSDomain);
    ms_opset->set_version(1);
  }

  auto* graph = model.mutable_graph();
  graph->set_name("telum_mini_transformer_block");

  // Graph IO.
  AddGraphInput(*graph, "X", {kBatch, kIn});
  AddGraphOutput(*graph, "Y", {kBatch, kHidden});

  // Initializers.
  // Weight W: [kIn, kHidden]
  const std::vector<float> W = {
      0.1f, -0.2f, 0.3f, 0.4f,
      -0.5f, 0.6f, -0.7f, 0.8f,
      0.9f, 1.0f, -1.1f, 1.2f,
  };
  AddFloatInitializer(*graph, "W", {kIn, kHidden}, W);

  // Bias B: [kHidden]
  const std::vector<float> B = {0.01f, -0.02f, 0.03f, 0.04f};
  AddFloatInitializer(*graph, "B", {kHidden}, B);

  // LayerNorm scale/bias: [kHidden]
  const std::vector<float> LNScale = {1.0f, 0.5f, 2.0f, 1.5f};
  const std::vector<float> LNBias = {0.1f, -0.2f, 0.3f, -0.4f};
  AddFloatInitializer(*graph, "LNScale", {kHidden}, LNScale);
  AddFloatInitializer(*graph, "LNBias", {kHidden}, LNBias);

  // ValueInfos for intermediates to ensure static-shape metadata is present.
  AddValueInfo(*graph, "Z", {kBatch, kHidden});
  AddValueInfo(*graph, "G", {kBatch, kHidden});
  AddValueInfo(*graph, "S", {kBatch, kHidden});

  // Nodes:
  // 1) Gemm: Z = X*W + B
  {
    auto* n = graph->add_node();
    n->set_op_type("Gemm");
    n->set_domain("");  // ONNX
    n->add_input("X");
    n->add_input("W");
    n->add_input("B");
    n->add_output("Z");

    AddIntAttr(*n, "transA", 0);
    AddIntAttr(*n, "transB", 0);
    AddFloatAttr(*n, "alpha", 1.0f);
    AddFloatAttr(*n, "beta", 1.0f);
  }

  // 2) Gelu (MS domain): G = Gelu(Z)
  {
    auto* n = graph->add_node();
    n->set_op_type("Gelu");
    n->set_domain(onnxruntime::kMSDomain);
    n->add_input("Z");
    n->add_output("G");
  }

  // 3) Softmax (ONNX): S = Softmax(G, axis=-1)
  {
    auto* n = graph->add_node();
    n->set_op_type("Softmax");
    n->set_domain("");  // ONNX
    n->add_input("G");
    n->add_output("S");
    AddIntAttr(*n, "axis", -1);
  }

  // 4) LayerNormalization (ONNX): Y = LN(S, LNScale, LNBias)
  {
    auto* n = graph->add_node();
    n->set_op_type("LayerNormalization");
    n->set_domain("");  // ONNX
    n->add_input("S");
    n->add_input("LNScale");
    n->add_input("LNBias");
    n->add_output("Y");  // omit optional Mean/InvStdDev outputs

    AddIntAttr(*n, "axis", -1);
    AddFloatAttr(*n, "epsilon", 1e-5f);
  }

  return model;
}

}  // namespace

TEST_F(TelumEndToEndTest, MiniTransformerBlock_GemmGeluSoftmaxLayerNorm) {
  ModelProto model = CreateMiniTransformerBlockModel();

  std::string bytes;
  ASSERT_TRUE(model.SerializeToString(&bytes));

  const gsl::span<const std::byte> model_data{reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()};

  // Inputs.
  const std::vector<float> x = {
      0.25f, -0.5f, 1.0f,
      -1.5f, 0.75f, 0.0f,
  };

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", CreateInputOrtValueOnCPU<float>({kBatch, kIn}, x)));

  EPVerificationParams params;
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  params.fp32_abs_err = 1e-4f;

  RunAndVerifyOutputsWithEP(model_data, CurrentTestName(),
                            DefaultTelumExecutionProvider(),
                            feeds,
                            params);
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime
