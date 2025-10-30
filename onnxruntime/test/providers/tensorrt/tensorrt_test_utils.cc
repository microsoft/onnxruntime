// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_cxx_api.h"
#include "core/common/path_string.h"

#include <iostream>
#include <fstream>

#include <onnx/onnx_pb.h>
#include <onnx/defs/schema.h>
#include <onnx/checker.h>

using namespace ONNX_NAMESPACE;

#define ONNX_IR_VERSION 11
#define OPSET_VERSION 23

namespace onnxruntime {
namespace test {

// Helper: make TensorProto
TensorProto MakeTensor(const std::string& name,
                       TensorProto::DataType dtype,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& vals) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(dtype);
  for (auto d : dims) t.add_dims(d);
  for (auto v : vals) t.add_int64_data(v);
  return t;
}

TensorProto MakeTensorFloat(const std::string& name,
                            const std::vector<int64_t>& dims,
                            const std::vector<float>& vals) {
  TensorProto t;
  t.set_name(name);
  t.set_data_type(TensorProto::FLOAT);
  for (auto d : dims) t.add_dims(d);
  for (auto v : vals) t.add_float_data(v);
  return t;
}

OrtStatus* CreateModelWithNodeOutputNotUsed(const PathString& model_name) {
  // --------------------
  // Create Model
  // --------------------
  ModelProto model;
  model.set_ir_version(ONNX_IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");  // empty = default ONNX domain
  opset->set_version(OPSET_VERSION);

  GraphProto* graph = model.mutable_graph();
  graph->set_name("DropoutMatMulGraph");

  // --------------------
  // Create Inputs
  // X: [3, 2]
  // W: [2, 3]
  // --------------------
  {
    auto* x = graph->add_input();
    x->set_name("X");

    auto* type = x->mutable_type();
    auto* tt = type->mutable_tensor_type();
    tt->set_elem_type(TensorProto::FLOAT);

    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(2);
  }

  {
    auto* w = graph->add_input();
    w->set_name("W");

    auto* type = w->mutable_type();
    auto* tt = type->mutable_tensor_type();
    tt->set_elem_type(TensorProto::FLOAT);

    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_value(2);
    shape->add_dim()->set_dim_value(3);
  }

  // --------------------
  // Output Y: [2, 3]
  // --------------------
  {
    auto* x = graph->add_output();
    x->set_name("Y");

    auto* type = x->mutable_type();
    auto* tt = type->mutable_tensor_type();
    tt->set_elem_type(TensorProto::FLOAT);

    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_value(2);
    shape->add_dim()->set_dim_value(3);
  }

  // --------------------
  // Dropout Node
  // --------------------
  {
    NodeProto* node = graph->add_node();
    node->set_name("DropoutNode");
    node->set_op_type("Dropout");

    node->add_input("X");
    node->add_output("dropout_out");
    node->add_output("dropout_mask");
  }

  // --------------------
  // MatMul Node
  // --------------------
  {
    NodeProto* node = graph->add_node();
    node->set_name("MatMulNode");
    node->set_op_type("MatMul");

    node->add_input("dropout_out");
    node->add_input("W");
    node->add_output("Y");
  }

  // --------------------
  // Validate
  // --------------------
  try {
    onnx::checker::check_model(model);
  } catch (const std::exception& ex) {
    std::string error_msg = "Model validation failed: " + std::string(ex.what());
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
  }

  std::ofstream ofs(model_name, std::ios::binary);
  if (!model.SerializeToOstream(&ofs)) {
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, "Failed to write model");
  }

  return nullptr;
}

OrtStatus* CreateModelWithTopKWhichContainsGraphOutput(const PathString& model_name) {
  ModelProto model;
  model.set_ir_version(ONNX_IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");   // empty = default ONNX domain
  opset->set_version(OPSET_VERSION);

  auto* graph = model.mutable_graph();
  graph->set_name("TopKGraph");

  // ======================
  // ---- Model Input ----
  // ======================
  {
    auto* inp = graph->add_input();
    inp->set_name("input");

    auto* type = inp->mutable_type();
    auto* tt = type->mutable_tensor_type();
    tt->set_elem_type(TensorProto::FLOAT);

    // Shape: ["N"]
    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_param("N");
  }

  // ======================
  // ---- Initializers ----
  // ======================
  {
    // K = [300]
    TensorProto K = MakeTensor("K", TensorProto::INT64, {1}, {300});
    *graph->add_initializer() = K;

    // zero = 0.0 (scalar)
    TensorProto zero = MakeTensor("zero", TensorProto::INT64, {}, {0});
    *graph->add_initializer() = zero;

    // twenty_six = 26 (scalar)
    TensorProto ts = MakeTensor("twenty_six", TensorProto::INT64, {}, {26});
    *graph->add_initializer() = ts;
  }

  // ======================
  // ---- TopK ----
  // ======================
  {
    NodeProto* n = graph->add_node();
    n->set_op_type("TopK");
    n->add_input("input");
    n->add_input("K");
    n->add_output("scores");
    n->add_output("topk_indices");
    n->set_name("TopK");
  }

  // ======================
  // ---- Less ----
  // ======================
  {
    NodeProto* n = graph->add_node();
    n->set_op_type("Less");
    n->add_input("topk_indices");
    n->add_input("zero");
    n->add_output("Less_output_0");
    n->set_name("Less");
  }

  // ======================
  // ---- Div ----
  // ======================
  {
    NodeProto* n = graph->add_node();
    n->set_op_type("Div");
    n->add_input("topk_indices");
    n->add_input("twenty_six");
    n->add_output("Div_17_output_0");
    n->set_name("Div");
  }

  // ======================
  // ---- Mod ----
  // ======================
  {
    NodeProto* n = graph->add_node();
    n->set_op_type("Mod");
    n->add_input("topk_indices");
    n->add_input("twenty_six");
    n->add_output("labels");
    n->set_name("Mod");
  }

  // =========================
  // ---- Graph Outputs ----
  // =========================
  auto add_output = [&](const std::string& name, TensorProto::DataType type, const std::string& dim) {
    auto* out = graph->add_output();
    out->set_name(name);

    auto* tt = out->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(type);

    auto* shape = tt->mutable_shape();
    shape->add_dim()->set_dim_param(dim);
  };

  add_output("scores", TensorProto::FLOAT, "K");
  add_output("Less_output_0", TensorProto::BOOL, "K");
  add_output("Div_17_output_0", TensorProto::INT64, "K");
  add_output("labels", TensorProto::INT64, "K");

  // ======================
  // Validate + Save
  // ======================
  try {
    onnx::checker::check_model(model);
  } catch (const std::exception& e) {
    std::string error_msg = "Model validation failed: " + std::string(e.what());
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
  }

  std::ofstream ofs(model_name, std::ios::binary);
  if (!model.SerializeToOstream(&ofs)) {
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, "Failed to write model");
  }

  return nullptr;
}
}
}
