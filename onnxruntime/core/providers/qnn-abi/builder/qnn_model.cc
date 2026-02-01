// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_model.h"

#include <iostream>
#include <fstream>
#include <gsl/gsl>
#include <thread>

#include "QnnOpDef.h"

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/builder/qnn_profile_serializer.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/qnn_allocator.h"
#include "core/providers/qnn-abi/qnn_ep_utils.h"
#include "core/providers/qnn-abi/shared_context.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Resolves the correct I/O order, preferring ONNX declaration order when available.
// When names don't match directly (e.g., with offload_graph_io_quantization), uses
// tensor_name_overrides_reversed to translate ONNX names to internal tensor names.
std::vector<std::string> ResolveGraphInputOutputOrder(
    const std::vector<std::string>* onnx_names,
    const std::unordered_set<std::string>& fused_names,
    const std::vector<std::string>& fused_order,
    const std::unordered_map<std::string, std::string>& tensor_name_overrides_reversed) {
  if (!onnx_names) {
    return fused_order;
  }

  bool names_match = std::all_of(onnx_names->begin(), onnx_names->end(),
                                 [&](const std::string& n) { return fused_names.count(n) > 0; });
  if (names_match) {
    return *onnx_names;
  }

  if (!tensor_name_overrides_reversed.empty()) {
    std::vector<std::string> mapped_order;
    for (const auto& onnx_name : *onnx_names) {
      auto it = tensor_name_overrides_reversed.find(onnx_name);
      if (it != tensor_name_overrides_reversed.end() && fused_names.count(it->second)) {
        mapped_order.push_back(it->second);
      }
    }
    if (mapped_order.size() == onnx_names->size()) {
      return mapped_order;
    }
  }

  return fused_order;
}

}  // namespace

bool QnnModel::GetGraphInfoFromModel(QnnModelWrapper& model_wrapper, const Ort::Logger& /* logger */) {
  bool rt = true;

  graph_info_ = std::make_unique<GraphInfo>(model_wrapper.GetQnnGraph(),
                                            model_wrapper.GetQnnGraphName(),
                                            model_wrapper.GetQnnGraphContext(),
                                            std::move(model_wrapper.GetGraphInputTensorWrappers()),
                                            std::move(model_wrapper.GetGraphOutputTensorWrappers()));

  return rt;
}

Ort::Status QnnModel::SetGraphInputOutputInfo(const QnnModelContext& context) {
  const OrtGraph& ort_graph = context.ort_graph;

  graph_inputs_.Clear();
  graph_outputs_.Clear();

  Ort::ConstNode fused_node{&context.fused_node};
  std::vector<Ort::ConstValueInfo> input_defs = fused_node.GetInputs();

  // Collect non-initializer inputs
  std::unordered_set<std::string> fused_input_names;
  std::vector<std::string> fused_input_order;
  for (const auto& input : input_defs) {
    std::string name = input.GetName();
    if (!IsConstantInitializer(ort_graph, name)) {
      fused_input_names.insert(name);
      fused_input_order.push_back(name);
    }
  }

  std::unordered_map<std::string, std::string> tensor_name_overrides_reversed;
  if (context.tensor_name_overrides) {
    for (const auto& [internal, onnx] : *context.tensor_name_overrides) {
      tensor_name_overrides_reversed[onnx] = internal;
    }
  }

  const std::vector<std::string> input_order = ResolveGraphInputOutputOrder(
      context.onnx_input_names, fused_input_names, fused_input_order, tensor_name_overrides_reversed);

  for (size_t idx = 0; idx < input_order.size(); ++idx) {
    const auto& name = input_order[idx];
    ORT_CXX_LOG(context.logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("input " + std::to_string(idx) + " " + name).c_str());
    graph_inputs_.indices.emplace(name, idx);
    graph_inputs_.names.push_back(name);
  }

  for (size_t i = 0; i < input_defs.size(); ++i) {
    const auto& value_info = input_defs[i];
    std::string name = value_info.GetName();
    if (IsConstantInitializer(ort_graph, name)) continue;

    auto shape_info = value_info.TypeInfo().GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType elem_type = shape_info.GetElementType();
    std::vector<int64_t> shape = shape_info.GetShape();

    for (const auto& s : shape) {
      RETURN_IF(s < 0, ("Dynamic shape is not supported yet, for input: " + name).c_str());
    }

    graph_inputs_.tensors.emplace(std::piecewise_construct,
                                  std::forward_as_tuple(name),
                                  std::forward_as_tuple(i, static_cast<int32_t>(elem_type), std::move(shape)));
  }

  std::vector<Ort::ConstValueInfo> output_defs = fused_node.GetOutputs();

  std::unordered_set<std::string> fused_output_names;
  std::vector<std::string> fused_output_order;
  for (const auto& output : output_defs) {
    std::string name = output.GetName();
    fused_output_names.insert(name);
    fused_output_order.push_back(name);
  }

  const std::vector<std::string> output_order = ResolveGraphInputOutputOrder(
      context.onnx_output_names, fused_output_names, fused_output_order, tensor_name_overrides_reversed);

  for (size_t idx = 0; idx < output_order.size(); ++idx) {
    const auto& name = output_order[idx];
    ORT_CXX_LOG(context.logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("output " + std::to_string(idx) + " " + name).c_str());
    graph_outputs_.indices.emplace(name, idx);
    graph_outputs_.names.push_back(name);
  }

  for (size_t i = 0; i < output_defs.size(); ++i) {
    const auto& value_info = output_defs[i];
    std::string name = value_info.GetName();

    auto shape_info = value_info.TypeInfo().GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType elem_type = shape_info.GetElementType();
    std::vector<int64_t> shape = shape_info.GetShape();

    for (const auto& s : shape) {
      RETURN_IF(s < 0, ("Dynamic shape is not supported yet, for output: " + name).c_str());
    }

    graph_outputs_.tensors.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(name),
                                   std::forward_as_tuple(i, static_cast<int32_t>(elem_type), std::move(shape)));
  }

  return Ort::Status();
}

const OrtNodeUnit& QnnModel::GetNodeUnit(const OrtNode* node,
                                         const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map) const {
  const auto node_unit_it = node_unit_map.find(node);
  if (node_unit_it == node_unit_map.end()) {
    ORT_CXX_API_THROW("Node does not have corresponding OrtNode.", ORT_EP_FAIL);
  }
  return *node_unit_it->second;
}

Ort::Status QnnModel::ComposeGraph(const QnnModelContext& context) {
  RETURN_IF(context.onnx_input_names == nullptr, "onnx_input_names is required for ComposeGraph");
  RETURN_IF(context.onnx_output_names == nullptr, "onnx_output_names is required for ComposeGraph");
  RETURN_IF(context.model_settings == nullptr, "model_settings is required for ComposeGraph");

  const OrtGraph& ort_graph = context.ort_graph;
  const OrtNode& fused_node = context.fused_node;
  const Ort::Logger& logger = context.logger;

  ORT_CXX_LOG(logger,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("ComposeGraph Graph name: " + Ort::ConstGraph(&ort_graph).GetName()).c_str());

  // Holder for the OrtNodes in the graph, this will guarantee the OrtNodes is
  // valid throughout the lifetime of the ModelBuilder
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;
  // GetQDQNodeUnits
  std::tie(node_unit_holder, node_unit_map) = GetAllOrtNodeUnits(api_ptrs_.ort_api, &ort_graph, logger);

  // This name must be same with the EPContext node name
  const auto& graph_name = Ort::ConstNode(&fused_node).GetName();
  RETURN_IF_ERROR(SetGraphInputOutputInfo(context));

  QnnModelWrapper qnn_model_wrapper = QnnModelWrapper(ort_graph, api_ptrs_, logger,
                                                      qnn_backend_manager_->GetQnnInterface(),
                                                      qnn_backend_manager_->GetQnnBackendHandle(),
                                                      graph_inputs_,
                                                      graph_outputs_,
                                                      qnn_backend_manager_->GetQnnBackendType(),
                                                      *context.model_settings,
                                                      context.tensor_name_overrides);

  qnn::profile::ProfilingInfo profiling_info;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  if (qnn_backend_manager_->ProfilingEnabled()) {
    profiling_info.graph_name = graph_name;
    profiling_info.start_time = qnn::utils::GetTimeStampInUs();
  }
#endif

  bool rt = qnn_model_wrapper.CreateQnnGraph(qnn_backend_manager_->GetQnnContext(), graph_name, context.graph_configs);

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  if (qnn_backend_manager_->ProfilingEnabled()) {
    profiling_info.stop_time = qnn::utils::GetTimeStampInUs();
    profiling_info.method_type = ProfilingMethodType::COMPOSE_GRAPHS;
  }
#endif

  RETURN_IF_NOT(rt, "Failed to initialize qnn_model_wrapper.");

  // NOTE: This function returns immediately when profiling is disabled.
  // Extracting profiling data can be expensive, but it is typically only enabled for debugging purposes
  // and not in production. We can improve synchronization for event profiling if it becomes an issue.
  RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo(profiling_info));

  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> qnn_node_groups;
  qnn_node_groups.reserve(node_unit_holder.size());

  RETURN_IF_ERROR(qnn::GetQnnNodeGroups(qnn_node_groups, qnn_model_wrapper, node_unit_map,
                                        node_unit_holder.size(), logger));

  for (const std::unique_ptr<qnn::IQnnNodeGroup>& qnn_node_group : qnn_node_groups) {
    Ort::Status status = qnn_node_group->AddToModelBuilder(qnn_model_wrapper, logger);

    if (!status.IsOK()) {
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_ERROR,
                  ("[QNN EP] Failed to add supported node to QNN graph during EP's compile call: " +
                   status.GetErrorMessage())
                      .c_str());
      return status;
    }
  }

  const bool build_json_graph = !context.json_qnn_graph_path.empty();
  RETURN_IF_NOT(qnn_model_wrapper.ComposeQnnGraph(build_json_graph), "Failed to compose Qnn graph.");

  LogTensorDetails(qnn_model_wrapper, graph_name, context.json_qnn_graph_path, logger);

  if (build_json_graph) {
    const nlohmann::json& json_graph = qnn_model_wrapper.GetQnnJSONGraph();
    std::ofstream ofs(context.json_qnn_graph_path);

    if (ofs.is_open()) {
      ofs << json_graph.dump();
      ofs.close();
    } else {
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_WARNING,
                  ("Could not open JSON graph file: " + context.json_qnn_graph_path).c_str());
    }
  }

  RETURN_IF_NOT(GetGraphInfoFromModel(qnn_model_wrapper, logger), "GetGraphInfoFromModel failed.");
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "GetGraphInfoFromModel completed.");
  return Ort::Status();
}

Ort::Status QnnModel::FinalizeGraphs(const Ort::Logger& logger) {
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "FinalizeGraphs started.");

  qnn::profile::ProfilingInfo profiling_info;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  if (qnn_backend_manager_->ProfilingEnabled()) {
    profiling_info.start_time = qnn::utils::GetTimeStampInUs();
  }
#endif

  Qnn_ErrorHandle_t status = qnn_backend_manager_->GetQnnInterface().graphFinalize(graph_info_->Graph(),
                                                                                   qnn_backend_manager_->GetQnnProfileHandle(),
                                                                                   nullptr);

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  if (qnn_backend_manager_->ProfilingEnabled()) {
    profiling_info.stop_time = qnn::utils::GetTimeStampInUs();
    profiling_info.method_type = ProfilingMethodType::FINALIZE;
    profiling_info.graph_name = graph_info_->Name();
  }
#endif

  if (QNN_GRAPH_NO_ERROR != status) {
    std::ostringstream oss;
    oss << "Failed to finalize QNN graph. Error code: " << status;
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
    return MAKE_EP_FAIL("Failed to finalize QNN graph.");
  }

  // NOTE: This function returns immediately when profiling is disabled.
  // Extracting profiling data can be expensive, but it is typically only enabled for debugging purposes
  // and not in production. We can improve synchronization for event profiling if it becomes an issue.
  RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo(profiling_info));

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "FinalizeGraphs completed.");
  return Ort::Status();
}

Ort::Status QnnModel::SetupQnnInputOutput(const Ort::Logger& logger) {
  ORT_CXX_LOG(logger,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("Setting up QNN input/output for graph: " + graph_info_->Name()).c_str());

  auto result = SetupTensors(qnn_input_infos_, graph_info_->InputTensors());

  if (!result.IsOK()) {
    const std::string message = "Failed to setup QNN input tensors for graph: " + graph_info_->Name() + ". " +
                                result.GetErrorMessage();
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, message.c_str());
    return MAKE_EP_FAIL(message.c_str());
  }

  result = SetupTensors(qnn_output_infos_, graph_info_->OutputTensors(), false);
  if (!result.IsOK()) {
    const std::string message = "Failed to setup QNN output tensors for graph: " + graph_info_->Name() + ". " +
                                result.GetErrorMessage();
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, message.c_str());
    return MAKE_EP_FAIL(message.c_str());
  }

  return Ort::Status();
}

static Ort::Status BindQnnTensorMemoryToOrtValueMemory(const OrtApi& ort_api,
                                                       const Ort::Logger& logger,
                                                       QnnBackendManager& qnn_backend_manager,
                                                       const OrtMemoryInfo* ort_value_memory_info,
                                                       void* ort_value_data, uint32_t ort_value_data_size,
                                                       Qnn_ContextHandle_t qnn_context,
                                                       Qnn_Tensor_t& qnn_tensor) {
  // either set qnn_tensor memHandle or clientBuf
  OrtMemoryInfoDeviceType ort_value_memory_info_device_type;
  ort_api.MemoryInfoGetDeviceType(ort_value_memory_info, &ort_value_memory_info_device_type);
  OrtDeviceMemoryType ort_value_memory_info_device_memory_type = ort_api.MemoryInfoGetDeviceMemType(ort_value_memory_info);
  const bool uses_shared_memory = (ort_value_memory_info_device_type == OrtMemoryInfoDeviceType_CPU &&
                                   ort_value_memory_info_device_memory_type == OrtDeviceMemoryType_HOST_ACCESSIBLE);

  if (!uses_shared_memory) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Setting Qnn_Tensor_t clientBuf to ORT tensor memory.");
    SetQnnTensorMemType(qnn_tensor, QNN_TENSORMEMTYPE_RAW);
    SetQnnTensorClientBuf(qnn_tensor, ort_value_data, ort_value_data_size);
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Setting Qnn_Tensor_t memHandle to ORT tensor shared memory.");
    Qnn_MemHandle_t qnn_mem_handle{};
    RETURN_IF_ERROR(qnn_backend_manager.GetOrRegisterContextMemHandle(qnn_context, ort_value_data, qnn_tensor,
                                                                      qnn_mem_handle));
    SetQnnTensorMemType(qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
    SetQnnTensorMemHandle(qnn_tensor, qnn_mem_handle);
  }

  return Ort::Status();
}

Ort::Status QnnModel::ExecuteGraph(OrtKernelContext* context,
                                   const Ort::Logger& logger) {
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "QnnModel::ExecuteGraphs");
  size_t num_inputs;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetInputCount(context, &num_inputs));

  size_t num_outputs;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetOutputCount(context, &num_outputs));
  RETURN_IF_NOT(qnn_input_infos_.size() <= num_inputs, "Inconsistent input sizes");
  RETURN_IF_NOT(qnn_output_infos_.size() == num_outputs, "Inconsistent output sizes");

  using namespace qnn::utils;
  auto TensorDataSize = [&ort_api = api_ptrs_.ort_api](auto ort_tensor) -> size_t {
    OrtTensorTypeAndShapeInfo* tensor_type_and_shape = nullptr;
    OrtStatusPtr tensor_status = ort_api.GetTensorTypeAndShape(ort_tensor, &tensor_type_and_shape);
    if (tensor_status != nullptr) {
      return 0;  // Return 0 on error, will be handled by caller
    }
    size_t length;
    tensor_status = ort_api.GetTensorShapeElementCount(tensor_type_and_shape, &length);
    if (tensor_status != nullptr) {
      return 0;  // Return 0 on error, will be handled by caller
    }
    ONNXTensorElementDataType element_type;
    tensor_status = ort_api.GetTensorElementType(tensor_type_and_shape, &element_type);
    if (tensor_status != nullptr) {
      return 0;  // Return 0 on error, will be handled by caller
    }
    size_t element_size = GetElementSizeByType(element_type);
    return element_size * length;
  };

  std::vector<Qnn_Tensor_t> qnn_inputs;
  qnn_inputs.reserve(qnn_input_infos_.size());

  for (const auto& qnn_input_info : qnn_input_infos_) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("model_input = " + qnn_input_info.tensor_wrapper->GetName() +
                 " index = " + std::to_string(qnn_input_info.ort_index))
                    .c_str());
    const OrtValue* ort_input_tensor = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetInput(context, qnn_input_info.ort_index, &ort_input_tensor));
    auto ort_tensor_size = TensorDataSize(ort_input_tensor);
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Qnn tensor size: " + std::to_string(qnn_input_info.tensor_byte_size) +
                 " Ort tensor size: " + std::to_string(ort_tensor_size))
                    .c_str());
    RETURN_IF_NOT(qnn_input_info.tensor_byte_size == ort_tensor_size,
                  "ORT Tensor data size does not match QNN tensor data size.");

    qnn_inputs.push_back(qnn_input_info.tensor_wrapper->GetQnnTensor());

    const OrtMemoryInfo* input_tensor_mem_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorMemoryInfo(ort_input_tensor, &input_tensor_mem_info));

    const void* raw_data;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorData(ort_input_tensor, &raw_data));

    RETURN_IF_ERROR(BindQnnTensorMemoryToOrtValueMemory(
        api_ptrs_.ort_api,
        logger,
        *qnn_backend_manager_,
        static_cast<const OrtMemoryInfo*>(input_tensor_mem_info),
        const_cast<void*>(raw_data), qnn_input_info.tensor_byte_size,
        graph_info_->GraphContext(),
        qnn_inputs.back()));
  }

  std::vector<Qnn_Tensor_t> qnn_outputs;
  qnn_outputs.reserve(qnn_output_infos_.size());

  for (auto& qnn_output_info : qnn_output_infos_) {
    const std::string& model_output_name = qnn_output_info.tensor_wrapper->GetName();
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("model_output = " + model_output_name +
                 " index = " + std::to_string(qnn_output_info.ort_index))
                    .c_str());
    const auto& ort_output_info = GetOutputInfo(model_output_name);
    const std::vector<int64_t>& output_shape = ort_output_info->shape_;
    OrtValue* ort_output_tensor = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetOutput(context,
                                                                         qnn_output_info.ort_index,
                                                                         output_shape.data(),
                                                                         output_shape.size(),
                                                                         &ort_output_tensor));

    auto ort_tensor_size = TensorDataSize(ort_output_tensor);
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Qnn tensor size: " + std::to_string(qnn_output_info.tensor_byte_size) +
                 " Ort tensor size: " + std::to_string(ort_tensor_size))
                    .c_str());
    RETURN_IF_NOT(qnn_output_info.tensor_byte_size == ort_tensor_size,
                  "ORT Tensor data size does not match QNN tensor data size");

    qnn_outputs.push_back(qnn_output_info.tensor_wrapper->GetQnnTensor());

    const OrtMemoryInfo* output_tensor_mem_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorMemoryInfo(ort_output_tensor, &output_tensor_mem_info));

    void* mutable_data;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorMutableData(ort_output_tensor, &mutable_data));

    RETURN_IF_ERROR(BindQnnTensorMemoryToOrtValueMemory(
        api_ptrs_.ort_api,
        logger,
        *qnn_backend_manager_,
        static_cast<const OrtMemoryInfo*>(output_tensor_mem_info),
        mutable_data, qnn_output_info.tensor_byte_size,
        graph_info_->GraphContext(),
        qnn_outputs.back()));
  }

  Qnn_ErrorHandle_t execute_status = QNN_GRAPH_NO_ERROR;
  {
    const auto& qnn_interface = qnn_backend_manager_->GetQnnInterface();

    // Acquire mutex before calling QNN APIs to support calling session.Run() from multiple threads.
    std::lock_guard<std::mutex> lock(graph_exec_mutex_);

    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Start execute QNN graph:" + graph_info_->Name()).c_str());

    qnn::profile::ProfilingInfo profiling_info;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (qnn_backend_manager_->ProfilingEnabled()) {
      profiling_info.start_time = qnn::utils::GetTimeStampInUs();
    }
#endif
    auto profile_backend_handle = qnn_backend_manager_->GetQnnProfileHandle();

    auto thread_id = std::this_thread::get_id();
    RETURN_IF_ERROR(qnn_backend_manager_->SetPerThreadHtpPowerConfigs(thread_id, true));

    execute_status = qnn_interface.graphExecute(graph_info_->Graph(),
                                                qnn_inputs.data(),
                                                static_cast<uint32_t>(qnn_inputs.size()),
                                                qnn_outputs.data(),
                                                static_cast<uint32_t>(qnn_outputs.size()),
                                                profile_backend_handle,
                                                nullptr);

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (qnn_backend_manager_->ProfilingEnabled()) {
      profiling_info.stop_time = qnn::utils::GetTimeStampInUs();
      profiling_info.method_type = ProfilingMethodType::EXECUTE;
      profiling_info.graph_name = graph_info_->Name();
    }
#endif

    RETURN_IF_ERROR(qnn_backend_manager_->SetPerThreadHtpPowerConfigs(thread_id, false));

    // NOTE: This function returns immediately when profiling is disabled.
    // Extracting profiling data can be expensive, but it is typically only enabled for debugging purposes
    // and not in production. We can improve synchronization for event profiling if it becomes an issue.
    RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo(profiling_info));
  }

  if (QNN_COMMON_ERROR_SYSTEM_COMMUNICATION == execute_status) {
    auto error_message = "NPU crashed. SSR detected. Caused QNN graph execute error. Error code: ";
    std::ostringstream oss;
    oss << error_message << execute_status;
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
    return MAKE_EP_FAIL(oss.str().c_str());
  }

  if (QNN_GRAPH_NO_ERROR != execute_status) {
    return MAKE_EP_FAIL(("QNN graph execute error. Error code: " + std::to_string(execute_status)).c_str());
  }

  return Ort::Status();
}

// Setup information for Qnn inputs/outputs used during execution.
Ort::Status QnnModel::SetupTensors(std::vector<QnnTensorInfo>& qnn_tensor_infos,
                                   const std::vector<QnnTensorWrapper>& tensor_wrappers,
                                   bool is_input) {
  size_t tensor_count = tensor_wrappers.size();
  RETURN_IF(0 == tensor_count, "Zero tensor size!");
  if (is_input) {
    auto input_count = graph_inputs_.indices.size();
    RETURN_IF(input_count < tensor_count, "The count of graph inputs should be at least the count of tensor_wrapper!");
    qnn_tensor_infos.resize(input_count);
  } else {
    qnn_tensor_infos.resize(tensor_count);
  }

  for (auto& tensor_wrapper : tensor_wrappers) {
    RETURN_IF(utils::QnnTensorHasDynamicShape(tensor_wrapper.GetQnnTensor()),
              ("QNN tensor (" + tensor_wrapper.GetName() + ") has dynamic shape. This is not supported yet.").c_str());

    const size_t length = utils::GetQnnTensorDataSizeInBytes(tensor_wrapper.GetTensorDims(),
                                                             tensor_wrapper.GetTensorDataType());
    const auto& tensor_name = tensor_wrapper.GetName();
    auto qnn_index = is_input ? GetGraphInputIndex(tensor_name) : GetOutputIndex(tensor_name);
    auto ort_index = is_input ? GetOrtInputIndex(tensor_name) : qnn_index;

    QnnTensorInfo& qnn_tensor_info = qnn_tensor_infos[qnn_index];
    qnn_tensor_info.tensor_wrapper = &tensor_wrapper;
    qnn_tensor_info.tensor_byte_size = static_cast<uint32_t>(length);
    qnn_tensor_info.ort_index = ort_index;
  }
  // The number of graph inputs and the number of tensor wrappers may not match.
  // - For example, for ResizeNearestNeighbor op, Qnn only cares about the 1st input,
  //   so the rest of the inputs are not converted to tensor wrappers.
  // - However, these remaining inputs still appear in the graph inputs, resulting in
  //   a discrepancy in the input quantities.
  // If not all inputs are used, erase the empty allocations in qnn_tensor_infos.
  if (is_input) {
    qnn_tensor_infos.erase(std::remove_if(qnn_tensor_infos.begin(),
                                          qnn_tensor_infos.end(),
                                          [](QnnTensorInfo qnn_tensor_info) { return qnn_tensor_info.tensor_wrapper == nullptr; }),
                           qnn_tensor_infos.end());
  }
  return Ort::Status();
}

void QnnModel::LogTensorDetails(QnnModelWrapper& qnn_model_wrapper,
                                const std::string& graph_name,
                                const std::string& json_qnn_graph_path,
                                const Ort::Logger& logger) const {
  // Only generate tensor details if we have a path to write to
  if (json_qnn_graph_path.empty()) {
    return;
  }

  // Helper lambda to convert Qnn_DataType_t to string
#define QNN_DATATYPE_CASE(type) \
  case type:                    \
    return #type

  auto QnnDataTypeToString = [](Qnn_DataType_t data_type) -> std::string_view {
    switch (data_type) {
      QNN_DATATYPE_CASE(QNN_DATATYPE_INT_8);
      QNN_DATATYPE_CASE(QNN_DATATYPE_INT_16);
      QNN_DATATYPE_CASE(QNN_DATATYPE_INT_32);
      QNN_DATATYPE_CASE(QNN_DATATYPE_INT_64);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UINT_8);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UINT_16);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UINT_32);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UINT_64);
      QNN_DATATYPE_CASE(QNN_DATATYPE_FLOAT_16);
      QNN_DATATYPE_CASE(QNN_DATATYPE_FLOAT_32);
      QNN_DATATYPE_CASE(QNN_DATATYPE_SFIXED_POINT_8);
      QNN_DATATYPE_CASE(QNN_DATATYPE_SFIXED_POINT_16);
      QNN_DATATYPE_CASE(QNN_DATATYPE_SFIXED_POINT_32);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UFIXED_POINT_8);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UFIXED_POINT_16);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UFIXED_POINT_32);
      QNN_DATATYPE_CASE(QNN_DATATYPE_BOOL_8);
      QNN_DATATYPE_CASE(QNN_DATATYPE_SFIXED_POINT_4);
      QNN_DATATYPE_CASE(QNN_DATATYPE_UFIXED_POINT_4);
      default:
        return "QNN_DATATYPE_UNDEFINED";
    }
  };

#undef QNN_DATATYPE_CASE

  // Build JSON log structure
  nlohmann::json tensor_log;
  tensor_log["graph_name"] = graph_name;
  tensor_log["inputs"] = nlohmann::json::array();
  tensor_log["initializers"] = nlohmann::json::array();

  size_t total_input_size = 0;
  size_t num_inputs = 0;
  size_t total_initializer_size = 0;
  size_t num_initializers = 0;

  // Collect input tensor information
  const Ort::ConstGraph graph(&qnn_model_wrapper.GetOrtGraph());
  for (const Ort::ConstValueInfo& input : graph.GetInputs()) {
    const std::string input_name = input.GetName();

    // Skip if it's an initializer
    if (qnn_model_wrapper.IsConstantInput(input_name)) {
      continue;
    }

    // Check if this tensor exists in the QNN model
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      const auto& tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_name);
      const auto& qnn_tensor = tensor_wrapper.GetQnnTensor();

      Qnn_DataType_t data_type = tensor_wrapper.GetTensorDataType();
      const auto& dims = tensor_wrapper.GetTensorDims();
      size_t size_bytes = utils::GetQnnTensorDataSizeInBytes(dims, data_type);
      uint32_t num_elements = CalcQnnTensorNumElems(qnn_tensor);

      nlohmann::json input_info;
      input_info["name"] = input_name;
      input_info["datatype"] = QnnDataTypeToString(data_type);
      input_info["num_elements"] = num_elements;
      input_info["size_bytes"] = size_bytes;

      tensor_log["inputs"].push_back(input_info);
      total_input_size += size_bytes;
      num_inputs++;
    }
  }

  // Build a map of initializer names to the operators that use them
  std::unordered_map<std::string, std::vector<std::string>> initializer_to_ops;

  for (const Ort::ConstNode& node : graph.GetNodes()) {
    if (static_cast<const OrtNode*>(node) == nullptr) {
      continue;
    }

    const std::string op_type = node.GetOperatorType();
    const std::string node_name = node.GetName();

    // Check each input of the node
    for (const Ort::ConstValueInfo& input : node.GetInputs()) {
      if (static_cast<const OrtValueInfo*>(input) == nullptr) {
        continue;
      }

      const std::string input_name = input.GetName();

      // Check if this input is an initializer
      if (qnn_model_wrapper.IsConstantInput(input_name)) {
        // Add this operator to the list of operators using this initializer
        std::string op_info = op_type + " (" + node_name + ")";
        initializer_to_ops[input_name].push_back(op_info);
      }
    }
  }

  // Collect initializer tensor information with operator usage
  for (const Ort::ConstValueInfo& initializer : graph.GetInitializers()) {
    const std::string initializer_name = initializer.GetName();

    // Check if this tensor exists in the QNN model
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(initializer_name)) {
      const auto& tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(initializer_name);
      const auto& qnn_tensor = tensor_wrapper.GetQnnTensor();

      Qnn_DataType_t data_type = tensor_wrapper.GetTensorDataType();
      const auto& dims = tensor_wrapper.GetTensorDims();
      size_t size_bytes = utils::GetQnnTensorDataSizeInBytes(dims, data_type);
      uint32_t num_elements = CalcQnnTensorNumElems(qnn_tensor);

      nlohmann::json init_info;
      init_info["name"] = initializer_name;
      init_info["datatype"] = QnnDataTypeToString(data_type);
      init_info["num_elements"] = num_elements;
      init_info["size_bytes"] = size_bytes;

      // Add operator information if available
      auto it = initializer_to_ops.find(initializer_name);
      if (it != initializer_to_ops.end() && !it->second.empty()) {
        init_info["used_by_operators"] = it->second;
      } else {
        init_info["used_by_operators"] = nlohmann::json::array();
      }

      tensor_log["initializers"].push_back(init_info);
      total_initializer_size += size_bytes;
      num_initializers++;
    }
  }

  // Add summary statistics
  tensor_log["summary"]["num_inputs"] = num_inputs;
  tensor_log["summary"]["total_input_size_bytes"] = total_input_size;
  tensor_log["summary"]["num_initializers"] = num_initializers;
  tensor_log["summary"]["total_initializer_size_bytes"] = total_initializer_size;
  tensor_log["summary"]["total_graph_size_bytes"] = total_input_size + total_initializer_size;
  tensor_log["summary"]["total_graph_size_mb"] = (total_input_size + total_initializer_size) / 1024.0 / 1024.0;

  // Write JSON log to file
  std::string tensor_log_path = json_qnn_graph_path;
  size_t ext_pos = tensor_log_path.find_last_of('.');
  if (ext_pos != std::string::npos) {
    tensor_log_path = tensor_log_path.substr(0, ext_pos) + "_tensor_log.json";
  } else {
    tensor_log_path += "_tensor_log.json";
  }

  std::ofstream tensor_log_file(tensor_log_path);
  if (tensor_log_file.is_open()) {
    tensor_log_file << tensor_log.dump(2);  // Pretty print with 2-space indentation
    tensor_log_file.close();
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO, ("Tensor log saved to: " + tensor_log_path).c_str());
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, ("Could not open tensor log file: " + tensor_log_path).c_str());
  }
}

Ort::Status QnnModel::DeserializeGraphInfoFromBinaryInfo(const QnnSystemContext_GraphInfo_t& qnn_sys_ctx_graph_info,
                                                         const Qnn_ContextHandle_t& context) {
  std::vector<QnnTensorWrapper> input_tensor_wrappers;
  std::vector<QnnTensorWrapper> output_tensor_wrappers;

  std::string graph_name;
  Qnn_Tensor_t* input_tensors = nullptr;
  Qnn_Tensor_t* output_tensors = nullptr;
  uint32_t graph_input_num = 0;
  uint32_t graph_output_num = 0;
  if (qnn_sys_ctx_graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    graph_name.assign(qnn_sys_ctx_graph_info.graphInfoV1.graphName);
    graph_input_num = qnn_sys_ctx_graph_info.graphInfoV1.numGraphInputs;
    graph_output_num = qnn_sys_ctx_graph_info.graphInfoV1.numGraphOutputs;

    input_tensors = qnn_sys_ctx_graph_info.graphInfoV1.graphInputs;
    output_tensors = qnn_sys_ctx_graph_info.graphInfoV1.graphOutputs;
  }
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 18)  // start from 2.25
  else if (qnn_sys_ctx_graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    graph_name.assign(qnn_sys_ctx_graph_info.graphInfoV2.graphName);
    graph_input_num = qnn_sys_ctx_graph_info.graphInfoV2.numGraphInputs;
    graph_output_num = qnn_sys_ctx_graph_info.graphInfoV2.numGraphOutputs;

    input_tensors = qnn_sys_ctx_graph_info.graphInfoV2.graphInputs;
    output_tensors = qnn_sys_ctx_graph_info.graphInfoV2.graphOutputs;
  }
#endif
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)  // start from 2.28
  else if (qnn_sys_ctx_graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
    graph_name.assign(qnn_sys_ctx_graph_info.graphInfoV3.graphName);
    graph_input_num = qnn_sys_ctx_graph_info.graphInfoV3.numGraphInputs;
    graph_output_num = qnn_sys_ctx_graph_info.graphInfoV3.numGraphOutputs;

    input_tensors = qnn_sys_ctx_graph_info.graphInfoV3.graphInputs;
    output_tensors = qnn_sys_ctx_graph_info.graphInfoV3.graphOutputs;
  }
#endif
  else {
    return MAKE_EP_FAIL("Unsupported context graph info version.");
  }
  RETURN_IF(nullptr == input_tensors, "Graph from cached context doesn't have any inputs.");
  RETURN_IF(nullptr == output_tensors, "Graph from cached context doesn't have any outputs.");

  // Copy graph input
  for (size_t i = 0; i < graph_input_num; ++i) {
    QnnTensorWrapper tensorwrapper;
    RETURN_IF_ERROR(tensorwrapper.Init(input_tensors[i]));
    input_tensor_wrappers.push_back(std::move(tensorwrapper));
  }
  // Copy graph output
  for (size_t i = 0; i < graph_output_num; ++i) {
    QnnTensorWrapper tensorwrapper;
    RETURN_IF_ERROR(tensorwrapper.Init(output_tensors[i]));
    output_tensor_wrappers.push_back(std::move(tensorwrapper));
  }

  Qnn_GraphHandle_t graph;
  auto qnn_interface = qnn_backend_manager_->GetQnnInterface();
  auto rt = qnn_interface.graphRetrieve(context, graph_name.c_str(), &graph);
  RETURN_IF(QNN_SUCCESS != rt, "Failed to retrieve QNN graph.");

  graph_info_ = std::make_unique<GraphInfo>(graph,
                                            graph_name,
                                            context,
                                            std::move(input_tensor_wrappers),
                                            std::move(output_tensor_wrappers));

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
