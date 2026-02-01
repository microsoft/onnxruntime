// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <vector>

#include "core/providers/qnn-abi/builder/qnn_def.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_backend_manager.h"
#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/rpcmem_library.h"

namespace onnxruntime {
namespace qnn {

struct QnnTensorInfo {
  const QnnTensorWrapper* tensor_wrapper = nullptr;
  uint32_t tensor_byte_size = 0;
  size_t ort_index = 0;
};

// Configuration for QnnModel::ComposeGraph and QnnModel::SetGraphInputOutputInfo.
struct QnnModelContext {
  const OrtGraph& ort_graph;
  const OrtNode& fused_node;
  const Ort::Logger& logger;

  // Names in ONNX declaration order, absent when loading from cached context.
  const std::vector<std::string>* onnx_input_names = nullptr;
  const std::vector<std::string>* onnx_output_names = nullptr;

  const ModelSettings* model_settings = nullptr;
  const QnnGraph_Config_t** graph_configs = nullptr;

  // Used by offload_graph_io_quantization to map internal QNN names to ONNX names.
  std::unordered_map<std::string, std::string>* tensor_name_overrides = nullptr;
  std::string json_qnn_graph_path;
};

class QnnModel {
 public:
  QnnModel(QnnBackendManager* qnn_backend_manager,
           const ApiPtrs& api_ptrs)
      : qnn_backend_manager_(qnn_backend_manager),
        api_ptrs_(ApiPtrs{api_ptrs.ort_api, api_ptrs.ep_api, api_ptrs.model_editor_api}) {
    qnn_backend_type_ = qnn_backend_manager_->GetQnnBackendType();
  }

  ~QnnModel() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnModel);

  Ort::Status ComposeGraph(const QnnModelContext& context);

  Ort::Status FinalizeGraphs(const Ort::Logger& logger);

  Ort::Status SetupQnnInputOutput(const Ort::Logger& logger);

  Ort::Status ExecuteGraph(OrtKernelContext* context, const Ort::Logger& logger);

  const OnnxTensorInfo* GetOutputInfo(const std::string& name) const {
    auto it = graph_outputs_.tensors.find(name);
    if (it == graph_outputs_.tensors.end()) {
      return nullptr;
    }
    return &(it->second);
  }

  Ort::Status SetGraphInputOutputInfo(const QnnModelContext& context);

  // Input index within ORT graph (includes initializers in count)
  size_t GetOrtInputIndex(const std::string& name) const {
    return GetInputOutputIndex(name, graph_inputs_.tensors);
  }

  // Input index excluding initializers
  size_t GetGraphInputIndex(const std::string& name) const {
    auto it = graph_inputs_.indices.find(name);
    if (it == graph_inputs_.indices.end()) {
      ORT_CXX_API_THROW("Input name not found.", ORT_EP_FAIL);
    }
    return it->second;
  }

  size_t GetOutputIndex(const std::string& name) const {
    return GetInputOutputIndex(name, graph_outputs_.tensors);
  }

  Ort::Status DeserializeGraphInfoFromBinaryInfo(const QnnSystemContext_GraphInfo_t& qnn_sys_ctx_graph_info,
                                                 const Qnn_ContextHandle_t& context);

  bool IsConstantInitializer(const OrtGraph& ort_graph,
                             const std::string& tensor_name) const {
    size_t num_initializers = 0;
    OrtStatus* status = api_ptrs_.ort_api.Graph_GetNumInitializers(&ort_graph, &num_initializers);
    if (status != nullptr) {
      return false;  // Return false on error
    }
    std::vector<const OrtValueInfo*> initializers(num_initializers);
    status = api_ptrs_.ort_api.Graph_GetInitializers(&ort_graph, initializers.data(), initializers.size());
    if (status != nullptr) {
      api_ptrs_.ort_api.ReleaseStatus(status);
      return false;
    }

    for (const OrtValueInfo* value_info : initializers) {
      const char* value_info_name = nullptr;
      status = api_ptrs_.ort_api.GetValueInfoName(value_info, &value_info_name);
      if (status != nullptr) {
        continue;  // Skip this initializer on error
      }

      if (std::string(value_info_name) == tensor_name) {
        bool is_constant_initializer = false;
        status = api_ptrs_.ort_api.ValueInfo_IsConstantInitializer(value_info, &is_constant_initializer);
        if (status != nullptr) {
          return false;  // Return false on error
        }
        return is_constant_initializer;
      }
    }
    return false;
  }

  const std::vector<std::string>& GetInputNames() const {
    return graph_inputs_.names;
  }

  const std::vector<std::string>& GetOutputNames() const {
    return graph_outputs_.names;
  }

  const std::unordered_map<std::string, OnnxTensorInfo>& GetInputsInfo() const {
    return graph_inputs_.tensors;
  }

  const std::unordered_map<std::string, OnnxTensorInfo>& GetOutputsInfo() const {
    return graph_outputs_.tensors;
  }

  const std::string& Name() const { return graph_info_->Name(); }

 private:
  const OrtNodeUnit& GetNodeUnit(const OrtNode* node,
                                 const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map) const;
  bool GetGraphInfoFromModel(QnnModelWrapper& model_wrapper, const Ort::Logger& logger);

  Ort::Status SetupTensors(std::vector<QnnTensorInfo>& tensors, const std::vector<QnnTensorWrapper>& tensor_wrappers,
                           bool is_input = true);

  void LogTensorDetails(QnnModelWrapper& qnn_model_wrapper,
                        const std::string& graph_name,
                        const std::string& json_qnn_graph_path,
                        const Ort::Logger& logger) const;

  QnnBackendType GetQnnBackendType() { return qnn_backend_type_; }

  size_t GetInputOutputIndex(const std::string& name, const std::unordered_map<std::string, OnnxTensorInfo>& io_info) const {
    auto it = io_info.find(name);
    if (it == io_info.end()) {
      ORT_CXX_API_THROW("Tensor name not found.", ORT_EP_FAIL);
    }
    return it->second.index_;
  }

 private:
  std::unique_ptr<GraphInfo> graph_info_;
  QnnBackendManager* qnn_backend_manager_ = nullptr;
  GraphInputOutputInfo graph_inputs_;
  GraphInputOutputInfo graph_outputs_;
  std::vector<QnnTensorInfo> qnn_input_infos_;
  std::vector<QnnTensorInfo> qnn_output_infos_;
  QnnBackendType qnn_backend_type_ = QnnBackendType::CPU;

  // Mutex acquired during graph execution to support multi-threaded inference of a single session.
  std::mutex graph_exec_mutex_;
  const ApiPtrs api_ptrs_;
};

}  // namespace qnn
}  // namespace onnxruntime
