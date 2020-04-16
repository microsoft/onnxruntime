// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include <inference_engine.hpp>
// IE defines a macro 'OPTIONAL' that conflicts the remaining headers using MSVC
#if defined(_MSC_VER)
#undef OPTIONAL
#endif

#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "backend_manager.h"
#include "ibackend.h"
#include "backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

InferenceEngine::Core BackendManager::ie_core_;

BackendManager::BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger,
                               std::string dev_id, std::string prec_str)
  : device_id_{dev_id}, precision_str_{prec_str} {
  if(precision_str_ == "FP32") {
    precision_ = InferenceEngine::Precision::FP32;
  } else if (precision_str_ == "FP16") {
    precision_ = InferenceEngine::Precision::FP16;
  } else {
    ORT_THROW("Invalid OpenVINO Precision type: " + precision_str_);
  }

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
  set_vpu_config_ = false;
  for (auto input : graph_inputs) {
    if(device_id_ == "MYRIAD"){
      auto shape = input->Shape();
      if(shape != nullptr){
        if(shape->dim_size() != 4){
          set_vpu_config_ = true;
        }
      }
    }
    auto it = inputdef_index_map.find(input->Name());
    if (it == inputdef_index_map.end()) {
      ORT_THROW("Input not found in the input defs list");
    }

    int index = it->second;
    input_indexes_.push_back(index);
  }

  auto graph_outputs_defs = fused_node->OutputDefs();
  i = 0;
  for (auto output_def : graph_outputs_defs){
    output_names_.insert({output_def->Name(), i});
    i++;
  }
  subgraph_name_ = fused_node->Name();
  model_proto_ = GetModelProtoFromFusedNode(fused_node, logger);

  if (ModelHasSymbolicInputDims(fused_node)) {
    LOGS_DEFAULT(INFO) <<
      "[OpenVINO-EP] Model has symbolic input dims. Defering backend initialization";
    has_dynamic_input_shape_ = true;
  } else {
    LOGS_DEFAULT(INFO) <<
      "[OpenVINO-EP] Model has concreate input dims. Initializing backend for graph " << subgraph_name_;
    has_dynamic_input_shape_ = false;
    concrete_backend_ = BackendFactory::MakeBackend(model_proto_, input_indexes_,
                                                    output_names_, device_id_,
                                                    precision_, ie_core_, subgraph_name_, set_vpu_config_);
  }
}

//Save ONNX Model
static common::Status SaveModel(ONNX_NAMESPACE::ModelProto& model_proto,
                                const std::string& file_path) {
  int fd;
  Status status = Env::Default().FileOpenWr(file_path, fd);
  google::protobuf::io::FileOutputStream output(fd);
  const bool result = model_proto.SerializeToZeroCopyStream(&output) && output.Flush();
  if (result)
    return Status::OK();
  else
    return Status::OK();
}

bool BackendManager::ModelHasSymbolicInputDims(const onnxruntime::Node* fused_node) const {
  bool has_sym_dims = false;
  auto graph_inputs = fused_node->GetFunctionBody()->Body().GetInputs();
  for (auto input : graph_inputs){
    if(input->Shape() == nullptr){
      has_sym_dims = true;
      break;
    }
    for (auto dim : input->Shape()->dim()) {
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

ONNX_NAMESPACE::ModelProto
BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node,
                                           const logging::Logger& logger) const {
  const auto* node_function = fused_node->GetFunctionBody();
  const std::string& name = fused_node->Name();
  ORT_ENFORCE(node_function != nullptr, "Could not extract function body for node: ", name);

  const onnxruntime::Graph& node_subgraph = node_function->Body();
  onnxruntime::Model model(node_subgraph.Name(), true, ModelMetaData{}, onnxruntime::ToPathString(""),
                           IOnnxRuntimeOpSchemaRegistryList{}, node_subgraph.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_subgraph.ToGraphProto();

  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    SaveModel(model_proto, name + ".onnx");
  }

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
                             std::string& device_id) {
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

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                      std::vector<std::vector<int64_t>> input_shapes) {
  auto model_copy = std::make_shared<ONNX_NAMESPACE::ModelProto>();
  std::string proto_str;
  model_proto.SerializeToString(&proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (size_t i = 0; i < input_shapes.size(); i++) {
    auto g_in_shape = graph_proto->mutable_input((int)i)->
                        mutable_type()->mutable_tensor_type()->mutable_shape();
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

    std::shared_ptr<IBackend> dynamic_backend;
    auto search = backend_map_.find(key);
    if (search == backend_map_.end()) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Creating concrete backend for key: " << key;
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_name_;
      auto modelproto_with_concrete_shapes = ReWriteInputShapeInfo(model_proto_, tensor_shapes);
      dynamic_backend = BackendFactory::MakeBackend(*modelproto_with_concrete_shapes,
                                                    input_indexes_,output_names_,
                                                    device_id_, precision_, ie_core_,subgraph_name_, set_vpu_config_);
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