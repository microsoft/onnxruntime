// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <core/common/safeint.h>

#include "model_builder.h"
#include "helper.h"
#include "op_builder_factory.h"

#include "core/providers/common.h"
#include "core/providers/coreml/model/model.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"

namespace onnxruntime {
namespace coreml {

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger, uint32_t coreml_flags)
    : graph_viewer_(graph_viewer),
      logger_(logger),
      coreml_flags_(coreml_flags) {
}

Status ModelBuilder::Initialize() {
  coreml_model_ = std::make_unique<CoreML::Specification::Model>();
  {  // initialize CoreML model
    // We support CorelML Specification Version 4 (Core ML 3)
    coreml_model_->set_specificationversion(4);
    auto* neural_network = coreml_model_->mutable_neuralnetwork();
    neural_network->set_arrayinputshapemapping(::CoreML::Specification::NeuralNetworkMultiArrayShapeMapping::EXACT_ARRAY_MAPPING);
  }

  PreprocessInitializers();
  ORT_RETURN_IF_ERROR(RegisterInitializers());
  ORT_RETURN_IF_ERROR(RegisterModelInputs());
  ORT_RETURN_IF_ERROR(AddOperations());
  ORT_RETURN_IF_ERROR(RegisterModelOutputs());

  return Status::OK();
}

/* static */ const IOpBuilder* ModelBuilder::GetOpBuilder(const Node& node) {
  const auto& op_builders = GetOpBuilders();
  const auto it = op_builders.find(node.OpType());
  if (it != op_builders.cend())
    return it->second;

  return nullptr;
}

void ModelBuilder::PreprocessInitializers() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      op_builder->AddInitializersToSkip(*this, *node);
    }
  }
}

Status ModelBuilder::RegisterInitializers() {
  for (const auto& pair : GetInitializerTensors()) {
    const auto& tensor = *pair.second;
    const auto& name = tensor.name();
    if (Contains(skipped_initializers_, name))
      continue;

    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = std::make_unique<COREML_SPEC::NeuralNetworkLayer>();
    layer->set_name(GetUniqueName("initializer_" + name));

    // TODO,look at using LoadConstantLayer instead of LoadConstantNDLayer
    auto* constant_tensor = layer->mutable_loadconstantnd();
    const auto& shape = tensor.dims();
    if (shape.empty()) {
      // This is a scalar initializer, CoreML constant layer requires a shape, make this a {1} tensor
      constant_tensor->mutable_shape()->Add(1);
    } else {
      std::transform(shape.cbegin(), shape.cend(),
                     google::protobuf::RepeatedFieldBackInserter(constant_tensor->mutable_shape()),
                     [](int64_t dim) -> uint64_t { return SafeInt<uint64_t>(dim); });
    }

    CreateCoreMLWeight(*constant_tensor->mutable_data(), tensor);
    *layer->mutable_output()->Add() = name;
    AddLayer(std::move(layer));
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) {
  const auto& name = node_arg.Name();
  const std::string input_output_type = is_input ? "input" : "output";

  if (is_input) {
    // input should not be an initializer
    if (Contains(GetInitializerTensors(), name))
      return Status::OK();

    // This input will not be used
    if (Contains(skipped_inputs_, name))
      return Status::OK();
  }

  auto* model_description = coreml_model_->mutable_description();
  auto& input_output = is_input
                           ? *model_description->mutable_input()->Add()
                           : *model_description->mutable_output()->Add();

  input_output.set_name(name);
  auto* multi_array = input_output.mutable_type()->mutable_multiarraytype();
  std::vector<int64_t> shape;

  {  // input_output shape
    const auto* shape_proto = node_arg.Shape();
    ORT_RETURN_IF(shape_proto == nullptr,
                  "shape_proto cannot be null for ", input_output_type, ": ", name);
    const auto& dims = shape_proto->dim();
    if (dims.empty()) {
      // If we have an empty shape, this is a scalar input,
      // Since all the input output of CoreML EP is MultiArray, we will make the scalar input output as a {1} MultiArray
      shape.push_back(1);

      // we need to change the shapes of these scalar outputs back to {} when CoreML EP returns these values to ORT
      if (!is_input) {
        AddScalarOutput(name);
      }
    } else {
      shape.reserve(dims.size());
      for (const auto& dim : dims) {
        ORT_RETURN_IF_NOT(dim.has_dim_value(),
                          "Dynamic shape is not supported yet, for ", input_output_type, ": ", name);
        shape.push_back(dim.dim_value());
      }
    }
  }

  *multi_array->mutable_shape() = {shape.cbegin(), shape.cend()};

  int32_t data_type;
  {  // type
    const auto* type_proto = node_arg.TypeAsProto();
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The  ", input_output_type, " of graph doesn't have elem_type: ", name);
    }

    data_type = type_proto->tensor_type().elem_type();
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        multi_array->set_datatype(COREML_SPEC::ArrayFeatureType::FLOAT32);
        break;
      default: {
        // TODO: support other type
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The  ", input_output_type, " of graph doesn't have valid type, name: ", name,
                               " type: ", type_proto->tensor_type().elem_type());
      }
    }
  }

  input_output_info_.emplace(name, OnnxTensorInfo{data_type, shape});

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputs() {
  for (const auto* node_arg : graph_viewer_.GetInputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, true /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::AddOperations() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(*this, *node, logger_));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Node [", node->Name(), "], type [", node->OpType(), "] is not supported");
    }
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelOutputs() {
  for (const auto* node_arg : graph_viewer_.GetOutputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, false /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::Compile(std::unique_ptr<Model>& model, const std::string& path) {
  ORT_RETURN_IF_ERROR(SaveCoreMLModel(path));
  model.reset(new Model(path, logger_, coreml_flags_));
  model->SetScalarOutputs(std::move(scalar_outputs_));
  model->SetInputOutputInfo(std::move(input_output_info_));
  return model->LoadModel();
}

Status ModelBuilder::SaveCoreMLModel(const std::string& path) {
  ORT_RETURN_IF_ERROR(Initialize());
  std::ofstream stream(path, std::ofstream::out | std::ofstream::binary);
  ORT_RETURN_IF_NOT(coreml_model_->SerializeToOstream(&stream), "Save the CoreML model failed");

  // TODO, Delete, debug only
  if (const char* path = std::getenv("ORT_COREML_EP_CONVERTED_MODEL_PATH")) {
    std::ofstream temp_stream(path, std::ofstream::out | std::ofstream::binary);
    ORT_RETURN_IF_NOT(coreml_model_->SerializeToOstream(&temp_stream), "Save the CoreML model failed");
  }

  return Status::OK();
}

void ModelBuilder::AddScalarOutput(const std::string& output_name) {
  scalar_outputs_.insert(output_name);
}

void ModelBuilder::AddLayer(std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer) {
  auto* neural_network = coreml_model_->mutable_neuralnetwork();
  neural_network->mutable_layers()->AddAllocated(layer.release());
}

void ModelBuilder::AddInitializerToSkip(const std::string& tensor_name) {
  skipped_initializers_.insert(tensor_name);
}

void ModelBuilder::AddInputToSkip(const std::string& input_name) {
  skipped_inputs_.insert(input_name);
}

std::string ModelBuilder::GetUniqueName(const std::string& base_name) {
  std::string unique_name;
  do {
    std::ostringstream os;
    os << base_name << "_token_" << name_token_++;
    unique_name = os.str();
  } while (Contains(unique_names_, unique_name));

  return unique_name;
}

}  // namespace coreml
}  // namespace onnxruntime
