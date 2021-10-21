// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include <inference_engine.hpp>
#include <fstream>

#include "contexts.h"
#include "backend_manager.h"
#include "ibackend.h"
#include "backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

static std::unique_ptr<GlobalContext> g_global_context;

GlobalContext& BackendManager::GetGlobalContext() {
  // This is not thread safe to call for the first time, but it is first called on the main thread by the constructor so it is safe.
  if (!g_global_context)
    g_global_context = std::make_unique<GlobalContext>();
  return *g_global_context;
}

void BackendManager::ReleaseGlobalContext() {
  g_global_context.reset();
}

BackendManager::BackendManager(const Node* fused_node, const logging::Logger& logger) {
  auto prec_str = GetGlobalContext().precision_str;
  if (prec_str == "FP32") {
    subgraph_context_.precision = InferenceEngine::Precision::FP32;
  } else if (prec_str == "FP16") {
    subgraph_context_.precision = InferenceEngine::Precision::FP16;
  } else {
    ORT_THROW("Invalid OpenVINO Precision type: " + prec_str);
  }

  // Save the indexes of graph inputs among fused_node's inputDefs
  // (which also contains initializers).
  auto node_input_defs = fused_node->InputDefs();
  int i = 0;
  for (auto idef : node_input_defs) {
    subgraph_context_.input_names.insert({idef->Name(), i});
    i++;
  }

  auto graph_inputs = fused_node->GetFunctionBody()->Body().GetInputs();
  for (auto input : graph_inputs) {
    if (GetGlobalContext().device_type.find("MYRIAD") != std::string::npos) {
      auto shape = input->Shape();
      if (shape != nullptr) {
        if (shape->dim_size() != 4) {
          subgraph_context_.set_vpu_config = true;
        }
      }
    }
    auto it = subgraph_context_.input_names.find(input->Name());
    if (it == subgraph_context_.input_names.end()) {
      ORT_THROW("Input not found in the input defs list");
    }
    int index = it->second;
    subgraph_context_.input_indexes.push_back(index);
  }

  auto graph_outputs_defs = fused_node->OutputDefs();
  i = 0;
  for (auto output_def : graph_outputs_defs) {
    subgraph_context_.output_names.insert({output_def->Name(), i});
    i++;
  }
  subgraph_context_.subgraph_name = fused_node->Name();
  model_proto_ = GetModelProtoFromFusedNode(fused_node, logger);

  if (ModelHasBatchedInputs(*model_proto_) &&
      GetGlobalContext().is_wholly_supported_graph &&
      GetGlobalContext().device_type.find("HDDL") != std::string::npos) {
    subgraph_context_.enable_batching = true;
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model can be Batch inferenced \n";
    auto model_copy = ReWriteBatchDimWithOne(*model_proto_);
    concrete_backend_ = BackendFactory::MakeBackend(*model_copy, GetGlobalContext(), subgraph_context_);
    subgraph_context_.has_dynamic_input_shape = false;

  } else if (ModelHasSymbolicInputDims(fused_node)) {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has symbolic input dims. Defering backend initialization";
    subgraph_context_.has_dynamic_input_shape = true;
  } else {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has concreate input dims. Initializing backend for graph " << subgraph_context_.subgraph_name;

    subgraph_context_.has_dynamic_input_shape = false;
    concrete_backend_ = BackendFactory::MakeBackend(*model_proto_, GetGlobalContext(), subgraph_context_);
  }
}

bool BackendManager::ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const {
  bool has_batched_inputs = true;

  for (int i = 0; i < (int)subgraph_context_.input_indexes.size(); i++) {
    auto& input = model_proto.graph().input(subgraph_context_.input_indexes[i]);

    // Batch-process only raw image inputs (NCHW or NHWC layouts)
    auto& shape = input.type().tensor_type().shape();
    if (shape.dim_size() != 4) {
      has_batched_inputs = false;
      break;
    }

    if (shape.dim(0).value_case() == shape.dim(0).kDimValue) {
      has_batched_inputs = false;
      break;
    }

    for (int index = 1; index < 4; index++) {
      if (shape.dim(index).value_case() != shape.dim(0).kDimValue) {
        has_batched_inputs = false;
        break;
      }
    }
    if (!has_batched_inputs) {
      break;
    }
  }
  return has_batched_inputs;
}

bool BackendManager::ModelHasSymbolicInputDims(const onnxruntime::Node* fused_node) const {
  bool has_sym_dims = false;
  auto graph_inputs = fused_node->GetFunctionBody()->Body().GetInputs();
  for (auto input : graph_inputs) {
    if (input->Shape() == nullptr) {
      has_sym_dims = true;
      break;
    }
    for (auto& dim : input->Shape()->dim()) {
      if (dim.value_case() != dim.kDimValue) {
        has_sym_dims = true;
        break;
      }
    }
    if (has_sym_dims) {
      break;
    }
  }
  return has_sym_dims;
}

std::unique_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node,
                                           const logging::Logger& logger) const {
  const auto* node_function = fused_node->GetFunctionBody();
  const std::string& name = fused_node->Name();
  ORT_ENFORCE(node_function != nullptr, "Could not extract function body for node: ", name);

  const onnxruntime::Graph& node_subgraph = node_function->Body();
  auto model = node_subgraph.CreateGraphViewer()->CreateModel(logger);

  auto model_proto = model->ToProto();
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *model_proto->mutable_graph() = *node_subgraph.ToGraphProto();

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::fstream dump(name + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }
#endif

  return model_proto;
}

std::vector<std::vector<int64_t>> GetInputTensorShapes(Ort::CustomOpApi& api,
                                                       OrtKernelContext* context) {
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t i = 0; i < api.KernelContext_GetInputCount(context); i++) {
    auto input_tensor = api.KernelContext_GetInput(context, i);
    auto tensor_info = api.GetTensorTypeAndShape(input_tensor);
    auto tensor_shape = api.GetTensorShape(tensor_info);
    input_shapes.push_back(tensor_shape);
    api.ReleaseTensorTypeAndShapeInfo(tensor_info);
  }
  return input_shapes;
}

std::string MakeMapKeyString(std::vector<std::vector<int64_t>>& shapes,
                             std::string& device_type) {
  std::string key;
  key += device_type;
  key += "|";  //separator
  for (auto shape : shapes) {
    for (auto dim : shape) {
      std::ostringstream o;
      o << dim;
      key += o.str();
      key += ",";
    }
    key += "|";
  }
  return key;
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                                      std::vector<std::vector<int64_t>> input_shapes) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (size_t i = 0; i < input_shapes.size(); i++) {
    auto g_in_shape = graph_proto->mutable_input((int)i)->mutable_type()->mutable_tensor_type()->mutable_shape();
    g_in_shape->clear_dim();
    auto shape = input_shapes[i];
    for (size_t dim = 0; dim < shape.size(); dim++) {
      g_in_shape->add_dim()->set_dim_value(shape[dim]);
    }
  }
  return model_copy;
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (int i = 0; i < graph_proto->input_size(); i++) {
    ONNX_NAMESPACE::TensorShapeProto* g_in_shape = graph_proto->mutable_input((int)i)->mutable_type()->mutable_tensor_type()->mutable_shape();
    g_in_shape->mutable_dim(0)->clear_dim_value();
    g_in_shape->mutable_dim(0)->set_dim_value(1);
  }
  return model_copy;
}

void BackendManager::Compute(Ort::CustomOpApi api, OrtKernelContext* context) {
  if (subgraph_context_.has_dynamic_input_shape) {
    std::vector<std::vector<int64_t>> tensor_shapes = GetInputTensorShapes(api, context);
    auto key = MakeMapKeyString(tensor_shapes, GetGlobalContext().device_type);

    if (GetGlobalContext().device_type.find("MYRIAD") != std::string::npos) {
      for (size_t i = 0; i < subgraph_context_.input_indexes.size(); i++) {
        if (tensor_shapes[i].size() != 4)
          subgraph_context_.set_vpu_config = true;
      }
    }

    std::shared_ptr<IBackend> dynamic_backend;
    auto search = backend_map_.find(key);
    if (search == backend_map_.end()) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Creating concrete backend for key: " << key;
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_context_.subgraph_name;
      auto modelproto_with_concrete_shapes = ReWriteInputShapeInfo(*model_proto_, tensor_shapes);
      dynamic_backend = BackendFactory::MakeBackend(*modelproto_with_concrete_shapes,
                                                    GetGlobalContext(), subgraph_context_);
      backend_map_.insert({key, dynamic_backend});
    } else {
      dynamic_backend = search->second;
    }

    dynamic_backend->Infer(api, context);
  } else {
    concrete_backend_->Infer(api, context);
  }
}

void BackendManager::ShutdownBackendManager() {
}

}  // namespace openvino_ep
}  // namespace onnxruntime
