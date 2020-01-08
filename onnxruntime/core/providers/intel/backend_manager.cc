// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <inference_engine.hpp>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "backend_manager.h"
#include "intel_graph.h"

namespace onnxruntime {
namespace intel_ep {

BackendManager::BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger) {
  device_id_ = "CPU";
  precision_ = InferenceEngine::Precision::FP32;
  std::string precision_str_ = "FP32";

#ifdef INTEL_CONFIG_CPU_FP32
#endif
#ifdef INTEL_CONFIG_GPU_FP32
  device_id_ = "GPU";
#endif
#ifdef INTEL_CONFIG_GPU_FP16
  device_id_ = "GPU";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str_ = "FP16";
#endif
#ifdef INTEL_CONFIG_MYRIAD
  device_id_ = "MYRIAD";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str_ = "FP16";
#endif
#ifdef INTEL_CONFIG_VAD_M
  device_id_ = "HDDL";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str_ = "FP16";
#endif

  // Save the indexes of graph inputs among fused_node's inputDefs
  // (which also contains initializers).
  std::map<std::string, int> inputdef_index_map;
  auto node_input_defs = fused_node->InputDefs();
  int i = 0;
  for (auto idef : node_input_defs) {
    inputdef_index_map.insert({idef->Name(), i});
    i++;
  }

  auto graph_inputs = fused_node->GetFunctionBody()->Body().GetInputs();
  for (auto input : graph_inputs) {
    auto it = inputdef_index_map.find(input->Name());
    if (it == inputdef_index_map.end()) {
      throw "Input not found in the input defs list";
    }

    int index = it->second;
    input_indexes_.push_back(index);
  }

  model_proto_ = GetModelProtoFromFusedNode(fused_node, logger);

  if (ModelHasSymbolicInputDims(model_proto_)) {
    LOGS_DEFAULT(INFO) << "[Intel-EP] Model has symbolic input dims. Defering backend initialization";
    has_dynamic_input_shape_ = true;
  } else {
    LOGS_DEFAULT(INFO) << "[Intel-EP] Model has concreate input dims. Initializing backend";
    has_dynamic_input_shape_ = false;
    concrete_backend_ = std::make_shared<IntelGraph>(model_proto_, input_indexes_, device_id_, precision_);
  }
}

//Save ONNX Model
static common::Status SaveModel(ONNX_NAMESPACE::ModelProto& model_proto, const std::string& file_path) {
  int fd;
  Status status = Env::Default().FileOpenWr(file_path, fd);

  google::protobuf::io::FileOutputStream output(fd);
  const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
  if (result)
    return Status::OK();
  else
    return Status::OK();
}

bool BackendManager::ModelHasSymbolicInputDims(const ONNX_NAMESPACE::ModelProto& model_proto) const {
  bool has_sym_dims = false;
  for (auto input : model_proto.graph().input()) {
    for (auto dim : input.type().tensor_type().shape().dim()) {
      if (dim.value_case() == dim.kDimParam) {
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

ONNX_NAMESPACE::ModelProto BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node, const logging::Logger& logger) const {
  const auto* node_function = fused_node->GetFunctionBody();

  ORT_ENFORCE(node_function != nullptr, "Could not extract function body for node: ", fused_node->Name());

  const onnxruntime::Graph& node_subgraph = node_function->Body();
  onnxruntime::Model model{node_subgraph.Name(), true, ModelMetaData{},
                           IOnnxRuntimeOpSchemaRegistryList{}, node_subgraph.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(), logger};

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_subgraph.ToGraphProto();

  if (intel_ep::IsDebugEnabled()) {
    SaveModel(model_proto, "intel_model.onnx");
  }

  return model_proto;
}

std::vector<std::vector<int64_t>> GetInputTensorShapes(Ort::CustomOpApi& api, OrtKernelContext* context) {
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

std::string MakeMapKeyString(std::vector<std::vector<int64_t>>& shapes, std::string& device_id) {
  std::string key;
  key += device_id;
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

std::shared_ptr<ONNX_NAMESPACE::ModelProto> ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                                                                  std::vector<std::vector<int64_t>> input_shapes) {
  auto model_copy = std::make_shared<ONNX_NAMESPACE::ModelProto>();
  std::string proto_str;
  model_proto.SerializeToString(&proto_str);
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

void BackendManager::Compute(Ort::CustomOpApi api, OrtKernelContext* context) {
  if (has_dynamic_input_shape_) {
    std::vector<std::vector<int64_t>> tensor_shapes = GetInputTensorShapes(api, context);
    auto key = MakeMapKeyString(tensor_shapes, device_id_);

    std::shared_ptr<IntelGraph> dynamic_backend;
    auto search = backend_map_.find(key);
    if (search == backend_map_.end()) {
      LOGS_DEFAULT(INFO) << "[Intel-EP] "
                         << "Creating concrete backend for key: " << key;
      auto modelproto_with_concrete_shapes = ReWriteInputShapeInfo(model_proto_, tensor_shapes);
      dynamic_backend = std::make_shared<IntelGraph>(*modelproto_with_concrete_shapes, input_indexes_, device_id_, precision_);
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

}  // namespace intel_ep
}  // namespace onnxruntime