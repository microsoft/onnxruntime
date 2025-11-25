// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_model.h"

#include <iostream>
#include <fstream>
#include <gsl/gsl>

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

bool QnnModel::GetGraphInfoFromModel(QnnModelWrapper& model_wrapper, const Ort::Logger& /* logger */) {
  bool rt = true;

  graph_info_ = std::make_unique<GraphInfo>(model_wrapper.GetQnnGraph(),
                                            model_wrapper.GetQnnGraphName(),
                                            model_wrapper.GetQnnGraphContext(),
                                            std::move(model_wrapper.GetGraphInputTensorWrappers()),
                                            std::move(model_wrapper.GetGraphOutputTensorWrappers()));

  return rt;
}

Ort::Status QnnModel::SetGraphInputOutputInfo(const OrtGraph& ort_graph,
                                              const OrtNode& fused_node,
                                              const Ort::Logger& logger) {
  size_t num_inputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Node_GetNumInputs(&fused_node, &num_inputs));
  std::vector<const OrtValueInfo*> input_defs(num_inputs);
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Node_GetInputs(&fused_node, input_defs.data(), input_defs.size()));
  RETURN_IF_ERROR(ParseGraphInputOrOutput(ort_graph, input_defs, input_names_, inputs_info_,
                                          model_input_index_map_, logger, true));

  size_t num_outputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Node_GetNumOutputs(&fused_node, &num_outputs));

  std::vector<const OrtValueInfo*> output_defs(num_outputs);
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Node_GetOutputs(&fused_node, output_defs.data(), output_defs.size()));
  RETURN_IF_ERROR(ParseGraphInputOrOutput(ort_graph, output_defs, output_names_, outputs_info_,
                                          model_output_index_map_, logger));

  return Ort::Status();
}

Ort::Status QnnModel::ParseGraphInputOrOutput(const OrtGraph& ort_graph,
                                              std::vector<const OrtValueInfo*> input_output_defs,
                                              std::vector<std::string>& input_output_names,
                                              std::unordered_map<std::string, OnnxTensorInfo>& input_output_info_table,
                                              std::unordered_map<std::string, size_t>& input_output_index_map,
                                              const Ort::Logger& logger,
                                              bool is_input) {
  for (size_t i = 0, end = input_output_defs.size(), index = 0; i < end; ++i) {
    const OrtValueInfo* input_output_def_data = input_output_defs[i];
    const char* input_output_def_name = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoName(input_output_def_data, &input_output_def_name));

    const auto name = std::string(input_output_def_name);
    if (is_input) {
      if (IsConstantInitializer(ort_graph, name)) {
        continue;  // exclude initializer inputs
      }
    }
    // Validate input/output shape
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ((is_input ? "input " : "output ") + std::to_string(i) + " " + name).c_str());
    input_output_index_map.emplace(name, index++);

    const OrtTypeInfo* type_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoTypeInfo(input_output_def_data, &type_info));

    const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape));

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorElementType(type_shape, &elem_type));

    size_t num_dims = 0;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetDimensionsCount(type_shape, &num_dims));

    std::vector<int64_t> shape;
    shape.resize(num_dims, 0);
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetDimensions(type_shape, shape.data(), shape.size()));
    for (const auto& s : shape) {
      RETURN_IF(s < 0, ("Dynamic shape is not supported yet, for output: " + name).c_str());
    }
    // use index i so that for graph input, it has initializers included
    input_output_info_table.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(name),
                                    std::forward_as_tuple(i, static_cast<int32_t>(elem_type), std::move(shape)));
    input_output_names.push_back(name);
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

Ort::Status QnnModel::ComposeGraph(const OrtGraph& ort_graph,
                                   const OrtNode& fused_node,
                                   const qnn::ModelSettings& model_settings,
                                   const Ort::Logger& logger,
                                   const QnnGraph_Config_t** graph_configs,
                                   const std::string& json_qnn_graph_path) {
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
  RETURN_IF_ERROR(SetGraphInputOutputInfo(ort_graph, fused_node, logger));

  QnnModelWrapper qnn_model_wrapper = QnnModelWrapper(ort_graph, api_ptrs_, logger,
                                                      qnn_backend_manager_->GetQnnInterface(),
                                                      qnn_backend_manager_->GetQnnBackendHandle(),
                                                      model_input_index_map_,
                                                      model_output_index_map_,
                                                      qnn_backend_manager_->GetQnnBackendType(),
                                                      model_settings);

  qnn::profile::ProfilingInfo profiling_info;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  if (qnn_backend_manager_->ProfilingEnabled()) {
    profiling_info.graph_name = graph_name;
    profiling_info.start_time = qnn::utils::GetTimeStampInUs();
  }
#endif

  bool rt = qnn_model_wrapper.CreateQnnGraph(qnn_backend_manager_->GetQnnContext(), graph_name, graph_configs);

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

  const bool build_json_graph = !json_qnn_graph_path.empty();
  RETURN_IF_NOT(qnn_model_wrapper.ComposeQnnGraph(build_json_graph), "Failed to compose Qnn graph.");

  if (build_json_graph) {
    const nlohmann::json& json_graph = qnn_model_wrapper.GetQnnJSONGraph();
    std::ofstream ofs(json_qnn_graph_path);

    if (ofs.is_open()) {
      ofs << json_graph.dump();
      ofs.close();
    } else {
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_WARNING,
                  ("Could not open JSON graph file: " + json_qnn_graph_path).c_str());
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
    const std::string message = "Failed to setup QNN input tensors for graph: " + graph_info_->Name();
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, message.c_str());
    return MAKE_EP_FAIL(message.c_str());
  }

  result = SetupTensors(qnn_output_infos_, graph_info_->OutputTensors(), false);
  if (!result.IsOK()) {
    const std::string message = "Failed to setup QNN output tensors for graph: " + graph_info_->Name();
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
    // Resize qnn_tensor_infos according to the number of graph inputs.
    auto input_count = GetGraphInputCount();
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
