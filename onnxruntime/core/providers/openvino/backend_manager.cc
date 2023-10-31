// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include <fstream>
#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "contexts.h"
#include "backend_manager.h"
#include "ibackend.h"
#include "backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

static std::unique_ptr<GlobalContext> g_global_context;

GlobalContext& BackendManager::GetGlobalContext() {
  // This is not thread safe to call for the first time,
  // but it is first called on the main thread by the constructor so it is safe.
  if (!g_global_context)
    g_global_context = std::make_unique<GlobalContext>();
  return *g_global_context;
}

void BackendManager::ReleaseGlobalContext() {
  g_global_context.reset();
}

BackendManager::BackendManager(const onnxruntime::Node& fused_node,
                               const onnxruntime::GraphViewer& subgraph,
                               const logging::Logger& logger) {
  auto prec_str = GetGlobalContext().precision_str;
  if (prec_str == "FP32") {
    subgraph_context_.precision = "FP32";
  } else if (prec_str == "FP16") {
    subgraph_context_.precision = "FP16";
  } else if (prec_str == "U8") {
    subgraph_context_.precision = "U8";
  } else {
    throw std::string("Invalid OpenVINO Precision type: " + prec_str);
  }

  // Save the indexes of graph inputs among fused_node's inputDefs
  // (which also contains initializers).
  auto node_input_defs = fused_node.InputDefs();
  int i = 0;
  for (auto idef : node_input_defs) {
    subgraph_context_.input_names.insert({idef->Name(), i});
    i++;
  }

  auto graph_inputs = subgraph.GetInputs();
  for (auto input : graph_inputs) {
    auto it = subgraph_context_.input_names.find(input->Name());
    if (it == subgraph_context_.input_names.end()) {
      throw std::string("Input not found in the input defs list");
    }
    int index = it->second;
    subgraph_context_.input_indexes.push_back(index);
  }

  auto graph_outputs_defs = fused_node.OutputDefs();
  i = 0;
  for (auto output_def : graph_outputs_defs) {
    subgraph_context_.output_names.insert({output_def->Name(), i});
    i++;
  }
  subgraph_context_.subgraph_name = fused_node.Name();
  model_proto_ = GetModelProtoFromFusedNode(fused_node, subgraph, logger);

  if (ModelHasSymbolicInputDims(subgraph)) {
    subgraph_context_.has_dynamic_input_shape = true;
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has symbolic input dims";
    if (GetGlobalContext().device_type.find("CPU") != std::string::npos ||
        GetGlobalContext().device_type.find("GPU") != std::string::npos) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Starting backend initialization. "
                         << "Creating backend Dynamic Shapes";
      try {
        concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                        GetGlobalContext(),
                                                        subgraph_context_);
      } catch (std::string const& msg) {
        throw msg;
      }
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_context_.subgraph_name;
    }
  } else {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has concrete input dims. "
                       << "Initializing backend for graph "
                       << subgraph_context_.subgraph_name;

    subgraph_context_.has_dynamic_input_shape = false;
    try {
      concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                      GetGlobalContext(),
                                                      subgraph_context_);
    } catch (std::string const& msg) {
      throw msg;
    }
  }
}

bool BackendManager::ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const {
  bool has_batched_inputs = true;

  for (int i = 0; i < static_cast<int>(subgraph_context_.input_indexes.size()); i++) {
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

bool BackendManager::ModelHasSymbolicInputDims(const onnxruntime::GraphViewer& subgraph) const {
  bool has_sym_dims = false;
  auto graph_inputs = subgraph.GetInputs();
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
BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node& fused_node,
                                           const onnxruntime::GraphViewer& subgraph,
                                           const logging::Logger& logger) const {
  auto model = subgraph.CreateModel(logger);

  auto model_proto = model->ToProto();
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  subgraph.ToProto(*model_proto->mutable_graph(), true, true);

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    const std::string& name = fused_node.Name();
    std::fstream dump(name + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }
#else
  ORT_UNUSED_PARAMETER(fused_node);
#endif

  return model_proto;
}

std::vector<std::vector<int64_t>> GetInputTensorShapes(Ort::KernelContext& context) {
  const auto input_count = context.GetInputCount();
  std::vector<std::vector<int64_t>> input_shapes;
  input_shapes.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto input_tensor = context.GetInput(i);
    auto tensor_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    input_shapes.push_back(std::move(tensor_shape));
  }
  return input_shapes;
}

std::string MakeMapKeyString(const std::vector<std::vector<int64_t>>& shapes,
                             const std::string& device_type) {
  std::string key;
  key += device_type;
  key += "|";  // separator
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
                                      const std::vector<std::vector<int64_t>>& input_shapes) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (size_t i = 0, limit = input_shapes.size(); i < limit; i++) {
    auto g_in_shape = graph_proto->mutable_input(static_cast<int>(i))
                          ->mutable_type()
                          ->mutable_tensor_type()
                          ->mutable_shape();
    g_in_shape->clear_dim();
    const auto& shape = input_shapes[i];
    for (size_t dim = 0, end = shape.size(); dim < end; dim++) {
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
    ONNX_NAMESPACE::TensorShapeProto* g_in_shape =
        graph_proto->mutable_input(static_cast<int>(i))
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape();
    g_in_shape->mutable_dim(0)->clear_dim_value();
    g_in_shape->mutable_dim(0)->set_dim_value(1);
  }
  return model_copy;
}

void BackendManager::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  std::chrono::high_resolution_clock::time_point start_compute, end_compute;
#ifdef OPENVINO_FIL_ENABLED
  static bool fil_enabled = true;
  if (fil_enabled) {
    start_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "Start Compute";
  }
#endif
  bool use_dynamic_backend = true;
  if (subgraph_context_.has_dynamic_input_shape &&
      (GetGlobalContext().device_type.find("CPU") != std::string::npos ||
       GetGlobalContext().device_type.find("GPU") != std::string::npos)) {
    concrete_backend_->Infer(context);
    use_dynamic_backend = false;
  } else if (use_dynamic_backend && subgraph_context_.has_dynamic_input_shape) {
    std::vector<std::vector<int64_t>> tensor_shapes = GetInputTensorShapes(ctx);
    auto key = MakeMapKeyString(tensor_shapes, GetGlobalContext().device_type);

    std::shared_ptr<IBackend> dynamic_backend;
    auto search = backend_map_.find(key);
    if (search == backend_map_.end()) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Creating concrete backend for key: " << key;
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_context_.subgraph_name;
      auto modelproto_with_concrete_shapes = ReWriteInputShapeInfo(*model_proto_, tensor_shapes);
      try {
        dynamic_backend = BackendFactory::MakeBackend(*modelproto_with_concrete_shapes,
                                                      GetGlobalContext(),
                                                      subgraph_context_);
      } catch (std::string const& msg) {
        throw msg;
      }
      backend_map_.insert({key, dynamic_backend});
    } else {
      dynamic_backend = search->second;
    }

    dynamic_backend->Infer(context);
  } else {
    concrete_backend_->Infer(context);
  }
#ifdef OPENVINO_FIL_ENABLED
  if (fil_enabled) {
    end_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "End Compute";
    std::chrono::duration<double> compute_time = end_compute - start_compute;
    std::cout << "Compute Time: " << compute_time.count() << " s" << std::endl;
    fil_enabled = false;  // calculating compute time for first run only
  }
#endif
}

void BackendManager::ShutdownBackendManager() {
}

}  // namespace openvino_ep
}  // namespace onnxruntime
