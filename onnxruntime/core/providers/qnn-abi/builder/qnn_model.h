// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <vector>

#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/builder/qnn_def.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_backend_manager.h"
#include "core/providers/qnn-abi/rpcmem_library.h"

namespace onnxruntime {
namespace qnn {

struct QnnTensorInfo {
  const QnnTensorWrapper* tensor_wrapper = nullptr;
  uint32_t tensor_byte_size = 0;
  size_t ort_index = 0;
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

  Status ComposeGraph(const OrtGraph& ort_graph,
                      const OrtNode& fused_node,
                      const qnn::ModelSettings& model_settings,
                      const logging::Logger& logger,
                      const QnnGraph_Config_t** graph_configs = nullptr,
                      const std::string& json_qnn_graph_path = "");

  Status FinalizeGraphs(const logging::Logger& logger);

  Status SetupQnnInputOutput(const logging::Logger& logger);

  Status ExecuteGraph(OrtKernelContext* context,
                      const logging::Logger& logger);

  const OnnxTensorInfo* GetOutputInfo(const std::string& name) const {
    auto it = outputs_info_.find(name);
    if (it == outputs_info_.end()) {
      LOGS_DEFAULT(ERROR) << "GetOutputInfo, output: " << name << "not exist!";
      return nullptr;
    }
    return &(it->second);
  }

  Status SetGraphInputOutputInfo(const OrtGraph& ort_graph,
                                 const OrtNode& fused_node,
                                 const logging::Logger& logger);
  Status ParseGraphInputOrOutput(const OrtGraph& ort_graph,
                                 std::vector<const OrtValueInfo*> input_output_defs,
                                 std::vector<std::string>& input_output_names,
                                 std::unordered_map<std::string, OnnxTensorInfo>& input_output_info_table,
                                 std::unordered_map<std::string, size_t>& input_output_index,
                                 const logging::Logger& logger,
                                 bool is_input = false);

  // Return the input index within Ort graph which has initializers included
  size_t GetOrtInputIndex(const std::string& name) const {
    return GetInputOutputIndex(name, inputs_info_);
  }

  // Return the pure input index which doesn't cover initializers
  size_t GetGraphInputIndex(const std::string& name) const {
    auto it = model_input_index_map_.find(name);
    ORT_ENFORCE(it != model_input_index_map_.end(), "Input name not found.");
    return it->second;
  }

  // Return the number of graph inputs
  size_t GetGraphInputCount() const {
    return model_input_index_map_.size();
  }

  size_t GetOutputIndex(const std::string& name) const {
    return GetInputOutputIndex(name, outputs_info_);
  }

  Status DeserializeGraphInfoFromBinaryInfo(const QnnSystemContext_GraphInfo_t& qnn_sys_ctx_graph_info,
                                            const Qnn_ContextHandle_t& context);

  bool IsConstantInitializer(const OrtGraph& ort_graph,
                             const std::string& tensor_name) const {
    size_t num_initializers = 0;
    OrtStatusPtr status = api_ptrs_.ort_api.Graph_GetNumInitializers(&ort_graph, &num_initializers);
    if (status != nullptr) {
      return false; // Return false on error
    }
    std::vector<const OrtValueInfo*> initializers(num_initializers);
    Status ort_status = ort_graph.GetInitializers(initializers);
    if (!ort_status.IsOK()) {
      return false;
    }

    for (const OrtValueInfo* value_info : initializers) {
      const char* value_info_name = nullptr;
      status = api_ptrs_.ort_api.GetValueInfoName(value_info, &value_info_name);
      if (status != nullptr) {
        continue; // Skip this initializer on error
      }

      if (std::string(value_info_name) == tensor_name) {
        bool is_constant_initializer = false;
        status = api_ptrs_.ort_api.ValueInfo_IsConstantInitializer(value_info, &is_constant_initializer);
        if (status != nullptr) {
          return false; // Return false on error
        }
        return is_constant_initializer;
      }
    }
    return false;
  }

  const std::vector<std::string>& GetInputNames() const {
    return input_names_;
  }

  const std::vector<std::string>& GetOutputNames() const {
    return output_names_;
  }

  const std::unordered_map<std::string, OnnxTensorInfo>& GetInputsInfo() const {
    return inputs_info_;
  }

  const std::unordered_map<std::string, OnnxTensorInfo>& GetOutputsInfo() const {
    return outputs_info_;
  }

  const std::string& Name() const { return graph_info_->Name(); }

 private:
  const OrtNodeUnit& GetNodeUnit(const OrtNode* node,
                                 const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map) const;
  bool GetGraphInfoFromModel(QnnModelWrapper& model_wrapper, const logging::Logger& logger);

  Status SetupTensors(std::vector<QnnTensorInfo>& tensors, const std::vector<QnnTensorWrapper>& tensor_wrappers,
                      bool is_input = true);

  QnnBackendType GetQnnBackendType() { return qnn_backend_type_; }

  size_t GetInputOutputIndex(const std::string& name, const std::unordered_map<std::string, OnnxTensorInfo>& io_info) const {
    auto it = io_info.find(name);
    ORT_ENFORCE(it != io_info.end(), "Input/Output name not found.");
    return it->second.index_;
  }

 private:
  std::unique_ptr<GraphInfo> graph_info_;
  QnnBackendManager* qnn_backend_manager_ = nullptr;
  // <input_name, input_index>, initializer inputs are excluded, keep the input index here
  std::unordered_map<std::string, size_t> model_input_index_map_;
  std::unordered_map<std::string, size_t> model_output_index_map_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, OnnxTensorInfo> inputs_info_;
  std::unordered_map<std::string, OnnxTensorInfo> outputs_info_;
  std::vector<QnnTensorInfo> qnn_input_infos_;
  std::vector<QnnTensorInfo> qnn_output_infos_;
  QnnBackendType qnn_backend_type_ = QnnBackendType::CPU;

  // Mutex acquired during graph execution to support multi-threaded inference of a single session.
  std::mutex graph_exec_mutex_;
  const ApiPtrs api_ptrs_;
};

}  // namespace qnn
}  // namespace onnxruntime
