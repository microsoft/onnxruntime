// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "core/common/status.h"
#include "QnnInterface.h"
#include "qnn_def.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/allocator.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper {
 public:
  QnnModelWrapper(const GraphViewer& graph_viewer,
                  const logging::Logger& logger,
                  const QNN_INTERFACE_VER_TYPE& qnn_interface,
                  const std::unordered_map<std::string, size_t>& input_index_map,
                  const std::unordered_map<std::string, size_t>& output_index_map,
                  const std::unordered_map<std::string, OnnxTensorInfo>& inputs_info,
                  const std::unordered_map<std::string, OnnxTensorInfo>& outputs_info,
                  const std::unordered_set<std::string>& initializer_lookup,
                  const onnxruntime::AllocatorPtr& cpu_allocator)
      : graph_viewer_(graph_viewer),
        logger_(logger),
        qnn_interface_(qnn_interface),
        input_index_map_(input_index_map),
        output_index_map_(output_index_map),
        initializer_lookup_(initializer_lookup),
        inputs_info_(inputs_info),
        outputs_info_(outputs_info),
        cpu_allocator_(cpu_allocator) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnModelWrapper);

  ~QnnModelWrapper() = default;

  bool Initialize(const Qnn_ContextHandle_t& context,
                  const char* graph_name,
                  bool debug = false,
                  const QnnGraph_Config_t** graph_configs = nullptr);

  size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

  bool AddQnnTensor(const std::string& node_name, const std::string& tensor_name, const Qnn_Tensor_t& qnn_tensor, bool is_param = false);

  bool AddTensor(const std::string& node_name, QnnTensorWrapper&& tensor_wrapper);

  bool GetQnnTensor(const std::string& tensor_name, Qnn_Tensor_t& tensor);

  bool AddNode(const std::string& name,
               const std::string& package_name,
               const std::string& type,
               std::vector<QnnParamWrapper>&& params,
               const std::vector<std::string>& input_names,
               std::vector<QnnTensorWrapper>&& output_wrappers,
               bool do_op_validation = false);

  bool AddParams(const std::string& node_name,
                 const std::vector<QnnParamWrapper>& param_wrappers);

  Qnn_GraphHandle_t GetQnnGraph() { return graph_; }

  std::string GetQnnGraphName() { return graph_name_; }

  std::vector<QnnTensorWrapper>&& GetGraphInputTensorWrappers() {
    return std::move(model_input_tensor_wrappers_);
  }

  std::vector<QnnTensorWrapper>&& GetGraphOutputTensorWrappers() {
    return std::move(model_output_tensor_wrappers_);
  }

  std::unordered_map<std::string, QnnTensorWrapper> GetModelTensorsMap() {
    return std::move(model_tensors_map_);
  }

  const InitializedTensorSet& GetInitializerTensors() const { return graph_viewer_.GetAllInitializedTensors(); }

  bool IsInitializerInput(std::string input_name) {
    if (initializer_lookup_.find(input_name) == initializer_lookup_.end()) {
      return false;
    }

    return true;
  }
  static bool GetOnnxShape(const NodeArg& node_arg, std::vector<uint32_t>& shape);

  bool ProcessOffset(const std::string& offset_name,
                     int32_t& offset_value);

  bool ProcessScale(const std::string& scale_name,
                    float& scale_value);

  bool ProcessQuantizationParameter(const std::optional<NodeUnitIODef::QuantParam>& quant_param,
                                    float& scale_value,
                                    int32_t& offset_value);

  bool QnnContainsTensor(const std::string& tensor_name) const;

  bool IsGraphOutput(const std::string& tensor_name) const {
    return output_index_map_.find(tensor_name) != output_index_map_.end();
  }

  bool IsGraphInput(const std::string& tensor_name) const {
    return input_index_map_.find(tensor_name) != input_index_map_.end();
  }

  onnxruntime::AllocatorPtr GetAllocator() const {
    if (cpu_allocator_ == nullptr) {
      LOGS_DEFAULT(ERROR) << "cpu_allocator is null!";
    }
    return cpu_allocator_;
  }

  Status AddTransposeNode(NodeIndex node_index,
                          const std::string& input_name,
                          const std::string& output_name,
                          const std::vector<uint32_t>& input_shape,
                          const std::vector<uint32_t>& transpose_perm,
                          const std::vector<uint32_t>& output_shape,
                          const Qnn_DataType_t& tensor_data_type,
                          const Qnn_QuantizeParams_t& quantize_param,
                          const bool is_for_input = true,
                          const bool is_for_output = false);

  // Tranpose NCHW->HWCN for QNN weight
  Status AddNchwToHwcnTranspose(NodeIndex node_index,
                                const std::string& input_name,
                                const std::string& output_name,
                                const std::vector<uint32_t>& input_shape,
                                const std::vector<uint32_t>& output_shape,
                                const Qnn_DataType_t& tensor_data_type,
                                const Qnn_QuantizeParams_t& quantize_param,
                                const bool is_for_input = true,
                                const bool is_for_output = false) {
    LOGS(logger_, VERBOSE) << "Add NCHW->HWCN Transpose node after Conv weight input: " << input_name
                           << " -> " << output_name;
    return AddTransposeNode(node_index, input_name, output_name, input_shape, nchw2hwcn_perm_, output_shape,
                            tensor_data_type, quantize_param, is_for_input, is_for_output);
  }

  // Tranpose CNHW->HWCN for QNN weight
  Status AddCnhwToHwcnTranspose(NodeIndex node_index,
                                const std::string& input_name,
                                const std::string& output_name,
                                const std::vector<uint32_t>& input_shape,
                                const std::vector<uint32_t>& output_shape,
                                const Qnn_DataType_t& tensor_data_type,
                                const Qnn_QuantizeParams_t& quantize_param,
                                const bool is_for_input = true,
                                const bool is_for_output = false) {
    LOGS(logger_, VERBOSE) << "Add CNHW->HWCN Transpose node after ConvTranspose weight input: " << input_name
                           << " -> " << output_name;
    return AddTransposeNode(node_index, input_name, output_name, input_shape, cnhw2hwcn_perm_, output_shape,
                            tensor_data_type, quantize_param, is_for_input, is_for_output);
  }

  const OnnxTensorInfo* TryGetModelInputInfo(const std::string& name) const {
    auto it = inputs_info_.find(name);
    if (it == inputs_info_.end()) {
      return nullptr;
    }
    return &(it->second);
  }

  const OnnxTensorInfo* TryGetModelOutputInfo(const std::string& name) const {
    auto it = outputs_info_.find(name);
    if (it == outputs_info_.end()) {
      return nullptr;
    }
    return &(it->second);
  }

 private:
  bool IsQDQNode(const Node& node) {
    if (node.OpType() == "QuantizeLinear" || node.OpType() == "DequantizeLinear") {
      return true;
    }
    return false;
  }

  void AddQuantizeNodeInfo(const std::string& input_name, const std::string& output_name) {
    auto pos = quantize_input_output_.find(input_name);
    if (pos != quantize_input_output_.end()) {
      return;
    }
    quantize_input_output_.emplace(input_name, output_name);
  }

  void AddDequantizeNodeInfo(const std::string& input_name, const std::string& output_name) {
    auto pos = dequantize_output_input_.find(output_name);
    if (pos != dequantize_output_input_.end()) {
      return;
    }
    dequantize_output_input_.emplace(output_name, input_name);
  }

  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;
  const QNN_INTERFACE_VER_TYPE& qnn_interface_;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string graph_name_ = "";
  bool debug_ = false;  // flag to indicate if requested graph is to be run in debug mode(i.e all
                        // intermediate tensors will be accessible to client)

  std::vector<QnnTensorWrapper> model_input_tensor_wrappers_;
  std::vector<QnnTensorWrapper> model_output_tensor_wrappers_;
  // keeps track of graph tensors to enable creating Qnn nodes from tensor names
  std::unordered_map<std::string, QnnTensorWrapper> model_tensors_map_;
  const std::unordered_map<std::string, size_t>& input_index_map_;
  const std::unordered_map<std::string, size_t>& output_index_map_;
  const std::unordered_set<std::string>& initializer_lookup_;
  const std::unordered_map<std::string, OnnxTensorInfo>& inputs_info_;
  const std::unordered_map<std::string, OnnxTensorInfo>& outputs_info_;
  // <input_name, output_name> for QNN Quantize node
  std::unordered_map<std::string, std::string> quantize_input_output_;
  // <output_name, input_name> for QNN Dequantize node
  std::unordered_map<std::string, std::string> dequantize_output_input_;
  onnxruntime::AllocatorPtr cpu_allocator_;
  const std::vector<uint32_t> nchw2nhwc_perm_{0, 2, 3, 1};
  const std::vector<uint32_t> nhwc2nchw_perm_{0, 3, 1, 2};
  const std::vector<uint32_t> nchw2hwcn_perm_{2, 3, 1, 0};
  const std::vector<uint32_t> cnhw2hwcn_perm_{2, 3, 0, 1};
};  // QnnModelWrapper

}  // namespace qnn
}  // namespace onnxruntime
