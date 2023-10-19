// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared/node_unit/node_unit.h"

namespace onnxruntime {
namespace qnn {

class QnnModel {
 public:
  QnnModel(const logging::Logger& logger,
           QnnBackendManager* qnn_backend_manager)
      : logger_(logger),
        qnn_backend_manager_(qnn_backend_manager) {
    qnn_backend_type_ = qnn_backend_manager_->GetQnnBackendType();
  }

  ~QnnModel() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnModel);

  Status ComposeGraph(const GraphViewer& graph_viewer,
                      const onnxruntime::Node& fused_node);

  Status FinalizeGraphs();

  Status SetupQnnInputOutput();

  Status ExecuteGraph(const Ort::KernelContext& context);

  const OnnxTensorInfo* GetOutputInfo(const std::string& name) const {
    auto it = outputs_info_.find(name);
    if (it == outputs_info_.end()) {
      LOGS_DEFAULT(ERROR) << "GetOutputInfo, output: " << name << "not exist!";
      return nullptr;
    }
    return &(it->second);
  }

  Status SetGraphInputOutputInfo(const GraphViewer& graph_viewer,
                                 const onnxruntime::Node& fused_node);
  Status ParseGraphInputOrOutput(ConstPointerContainer<std::vector<NodeArg*>>& input_output_defs,
                                 std::vector<std::string>& input_output_names,
                                 std::unordered_map<std::string, OnnxTensorInfo>& input_output_info_table,
                                 std::unordered_map<std::string, size_t>& input_output_index,
                                 bool is_input = false);

  const std::unordered_set<std::string>& GetInitializerInputs() const { return initializer_inputs_; }
  bool IsGraphInitializerInput(const std::string input_name) {
    return initializer_inputs_.find(input_name) != initializer_inputs_.end();
  }

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

  size_t GetOutputIndex(const std::string& name) const {
    return GetInputOutputIndex(name, outputs_info_);
  }

  Status DeserializeGraphInfoFromBinaryInfo(const QnnSystemContext_GraphInfo_t& qnn_sys_ctx_graph_info);

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

  const std::string& Name() { return graph_info_->Name(); }

 private:
  const NodeUnit& GetNodeUnit(const Node* node,
                              const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map) const;
  bool GetGraphInfoFromModel(QnnModelWrapper& model_wrapper);

  Status GetQnnTensorDataLength(const std::vector<uint32_t>& dims,
                                Qnn_DataType_t data_type,
                                size_t& data_length) const;

  Status SetupTensors(std::vector<Qnn_Tensor_t>& tensors, const std::vector<QnnTensorWrapper>& tensor_wrappers, bool is_input = true);

  QnnBackendType GetQnnBackendType() { return qnn_backend_type_; }

  size_t GetInputOutputIndex(const std::string& name, const std::unordered_map<std::string, OnnxTensorInfo>& io_info) const {
    auto it = io_info.find(name);
    ORT_ENFORCE(it != io_info.end(), "Input/Output name not found.");
    return it->second.index_;
  }

 private:
  const logging::Logger& logger_;
  std::unique_ptr<GraphInfo> graph_info_;
  QnnBackendManager* qnn_backend_manager_ = nullptr;
  // <input_name, input_index>, initializer inputs are excluded, keep the input index here
  std::unordered_map<std::string, size_t> model_input_index_map_;
  std::unordered_map<std::string, size_t> model_output_index_map_;
  // TODO: remove initializer_inputs_, use QnnModelWrapper
  std::unordered_set<std::string> initializer_inputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, OnnxTensorInfo> inputs_info_;
  std::unordered_map<std::string, OnnxTensorInfo> outputs_info_;
  std::vector<Qnn_Tensor_t> qnn_inputs_;
  std::vector<Qnn_Tensor_t> qnn_outputs_;
  QnnBackendType qnn_backend_type_ = QnnBackendType::CPU;
};

}  // namespace qnn
}  // namespace onnxruntime
