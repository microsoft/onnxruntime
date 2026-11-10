// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "stateful/diffusion/ddim.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace test {
namespace stateful {

inline void SetTensorType(ONNX_NAMESPACE::ValueInfoProto& value,
                          const char* name,
                          int element_type,
                          std::initializer_list<int64_t> dimensions) {
  value.set_name(name);
  auto* tensor_type = value.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(element_type);
  auto* shape = tensor_type->mutable_shape();
  for (int64_t dimension : dimensions) {
    auto* dim = shape->add_dim();
    if (dimension < 0) {
      dim->set_dim_param(std::string(name) + "_sequence");
    } else {
      dim->set_dim_value(dimension);
    }
  }
}

inline ONNX_NAMESPACE::ModelProto MakeModel(const char* name) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(18);
  model.mutable_graph()->set_name(name);
  return model;
}

inline ONNX_NAMESPACE::NodeProto& AddNode(
    ONNX_NAMESPACE::GraphProto& graph,
    const char* operation,
    std::initializer_list<const char*> inputs,
    const char* output) {
  auto* node = graph.add_node();
  node->set_op_type(operation);
  for (const char* input : inputs) {
    node->add_input(input);
  }
  node->add_output(output);
  return *node;
}

inline ONNX_NAMESPACE::ModelProto MakeFeedbackModel() {
  auto model = MakeModel("stateful_feedback_reference");
  auto& graph = *model.mutable_graph();
  SetTensorType(*graph.add_input(), "input_value", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  SetTensorType(*graph.add_input(), "past_value", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  SetTensorType(*graph.add_output(), "present_value", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  SetTensorType(*graph.add_output(), "done", ONNX_NAMESPACE::TensorProto_DataType_BOOL, {});

  auto* limit = graph.add_initializer();
  limit->set_name("limit");
  limit->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  limit->add_float_data(5.0f);

  AddNode(graph, "Add", {"input_value", "past_value"}, "present_value");
  AddNode(graph, "Greater", {"present_value", "limit"}, "done");
  return model;
}

inline ONNX_NAMESPACE::ModelProto MakeRenamedStageModel() {
  auto model = MakeModel("stateful_renamed_stage_reference");
  auto& graph = *model.mutable_graph();
  SetTensorType(*graph.add_input(), "upstream", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  SetTensorType(*graph.add_output(), "pipeline_result", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  AddNode(graph, "Add", {"upstream", "upstream"}, "pipeline_result");
  return model;
}

inline ONNX_NAMESPACE::ModelProto MakeDecoderModel() {
  auto model = MakeModel("stateful_decoder_reference");
  auto& graph = *model.mutable_graph();
  SetTensorType(*graph.add_input(), "token", ONNX_NAMESPACE::TensorProto_DataType_INT64, {1});
  SetTensorType(*graph.add_input(), "past_key", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});
  SetTensorType(*graph.add_output(), "logits", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 4});
  SetTensorType(*graph.add_output(), "present_key", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});

  auto* scores = graph.add_initializer();
  scores->set_name("scores");
  scores->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  scores->add_dims(4);
  scores->add_dims(4);
  for (float value : {0.f, 4.f, 1.f, -2.f, 0.f, 1.f, 5.f, -2.f,
                      -1.f, 1.f, 2.f, 6.f, 6.f, 2.f, 1.f, 0.f}) {
    scores->add_float_data(value);
  }
  auto& gather = AddNode(graph, "Gather", {"scores", "token"}, "logits");
  auto* gather_axis = gather.add_attribute();
  gather_axis->set_name("axis");
  gather_axis->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  gather_axis->set_i(0);

  auto& cast = AddNode(graph, "Cast", {"token"}, "key_value");
  auto* to = cast.add_attribute();
  to->set_name("to");
  to->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  to->set_i(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* shape = graph.add_initializer();
  shape->set_name("key_shape");
  shape->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape->add_dims(4);
  for (int64_t dimension : {1, 1, 1, 1}) {
    shape->add_int64_data(dimension);
  }
  AddNode(graph, "Reshape", {"key_value", "key_shape"}, "new_key");
  auto& concat = AddNode(graph, "Concat", {"past_key", "new_key"}, "present_key");
  auto* concat_axis = concat.add_attribute();
  concat_axis->set_name("axis");
  concat_axis->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  concat_axis->set_i(2);
  return model;
}

inline ONNX_NAMESPACE::ModelProto MakeDenoiserModel() {
  auto model = MakeModel("stateful_denoiser_reference");
  auto& graph = *model.mutable_graph();
  SetTensorType(*graph.add_input(), "latent", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, 2, 2});
  SetTensorType(*graph.add_input(), "timestep", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  SetTensorType(*graph.add_output(), "noise_prediction", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, 2, 2});
  AddNode(graph, "Add", {"latent", "timestep"}, "noise_prediction");
  return model;
}

inline DdimOptions MakeDdimOptions(int64_t steps = 3) {
  DdimOptions options;
  options.latent_shape = {1, 1, 1, 3};
  options.num_inference_steps = steps;
  options.guidance_scale = 7.5f;
  options.timestep_format = DdimTimestepFormat::Float32Scalar;
  options.latent_input = "sample";
  options.timestep_input = "time";
  options.conditioning_input = "conditioning";
  options.prediction_output = "epsilon";
  options.initialization_input = "init";
  options.positive_conditioning_input = "positive";
  options.negative_conditioning_input = "negative";
  options.latent_output = "latents";
  return options;
}

inline ONNX_NAMESPACE::ModelProto MakeDdimDenoiserModel(const DdimOptions& options) {
  auto model = MakeModel("stateful_ddim_denoiser");
  auto& graph = *model.mutable_graph();
  const auto& shape = options.latent_shape;
  const int64_t batch = shape[0] * (options.guidance_mode == DdimGuidanceMode::Batched ? 2 : 1);
  const bool integer = options.timestep_format == DdimTimestepFormat::Int64Scalar ||
                       options.timestep_format == DdimTimestepFormat::Int64Vector;
  const bool vector = options.timestep_format == DdimTimestepFormat::Int64Vector ||
                      options.timestep_format == DdimTimestepFormat::Float32Vector;
  SetTensorType(*graph.add_input(), options.latent_input.c_str(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                {batch, shape[1], shape[2], shape[3]});
  SetTensorType(*graph.add_input(), options.timestep_input.c_str(),
                integer ? ONNX_NAMESPACE::TensorProto_DataType_INT64 : ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                vector ? std::initializer_list<int64_t>{1} : std::initializer_list<int64_t>{});
  SetTensorType(*graph.add_input(), options.conditioning_input.c_str(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                {batch, 1, 1, 1});
  SetTensorType(*graph.add_output(), options.prediction_output.c_str(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                {batch, shape[1], shape[2], shape[3]});
  for (const auto& [name, value] : {std::pair{"latent_scale", 0.125f}, std::pair{"time_scale", 0.0001f}}) {
    auto* initializer = graph.add_initializer();
    initializer->set_name(name);
    initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    initializer->add_float_data(value);
  }
  std::string time = options.timestep_input;
  if (integer) {
    auto& cast = AddNode(graph, "Cast", {time.c_str()}, "float_timestep");
    auto* attribute = cast.add_attribute();
    attribute->set_name("to");
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attribute->set_i(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    time = "float_timestep";
  }
  AddNode(graph, "Mul", {options.latent_input.c_str(), "latent_scale"}, "scaled_latent");
  AddNode(graph, "Mul", {time.c_str(), "time_scale"}, "scaled_timestep");
  AddNode(graph, "Add", {"scaled_latent", "scaled_timestep"}, "time_prediction");
  AddNode(graph, "Add", {"time_prediction", options.conditioning_input.c_str()}, options.prediction_output.c_str());
  return model;
}

inline ONNX_NAMESPACE::ModelProto MakeDynamicDecoderModel() {
  auto model = MakeDecoderModel();
  auto& graph = *model.mutable_graph();
  graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  graph.mutable_output(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  SetTensorType(*graph.mutable_input(0), "token", ONNX_NAMESPACE::TensorProto_DataType_INT64, {1, -1});
  SetTensorType(*graph.mutable_output(0), "logits", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  for (auto& initializer : *graph.mutable_initializer()) {
    if (initializer.name() == "key_shape") initializer.set_int64_data(2, -1);
  }
  return model;
}

}  // namespace stateful
}  // namespace test
}  // namespace onnxruntime
