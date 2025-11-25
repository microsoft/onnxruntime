// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "QnnInterface.h"

#include "core/providers/qnn-abi/builder/qnn_def.h"
#include "core/providers/qnn-abi/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Stores information about an ONNX input or output tensor.
// Filled out by QnnModelWrapper::GetTensorInfo()
struct TensorInfo {
  std::vector<uint32_t> shape;
  Qnn_DataType_t qnn_data_type;
  QnnQuantParamsWrapper quant_param;
  bool is_initializer;
  const OrtValueInfo* initializer_tensor;
};

struct ModelSettings {
  bool offload_graph_io_quantization = false;
  bool htp_shared_memory = false;
};

class QnnModelWrapper {
 public:
  QnnModelWrapper(const OrtGraph& ort_graph,
                  const ApiPtrs& api_ptrs,
                  const Ort::Logger& logger,
                  const QNN_INTERFACE_VER_TYPE& qnn_interface,
                  const Qnn_BackendHandle_t& backend_handle,
                  const std::unordered_map<std::string, size_t>& input_index_map,
                  const std::unordered_map<std::string, size_t>& output_index_map,
                  QnnBackendType qnn_backend_type,
                  const ModelSettings& model_settings)
      : ort_graph_(ort_graph),
        logger_(logger),
        qnn_interface_(qnn_interface),
        backend_handle_(backend_handle),
        input_index_map_(input_index_map),
        output_index_map_(output_index_map),
        qnn_backend_type_(qnn_backend_type),
        model_settings_(model_settings),
        api_ptrs_(ApiPtrs{api_ptrs.ort_api, api_ptrs.ep_api, api_ptrs.model_editor_api}) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnModelWrapper);

  ~QnnModelWrapper() = default;

  const ModelSettings& GetModelSettings() const { return model_settings_; }

  bool CreateQnnGraph(const Qnn_ContextHandle_t& context,
                      const std::string& graph_name,
                      const QnnGraph_Config_t** graph_configs = nullptr);

  // Make a QnnTensorWrapper from an onnx input or output.
  Ort::Status MakeTensorWrapper(const OrtNodeUnitIODef& tensor, QnnTensorWrapper& tensor_wrapper) const;
  Ort::Status MakeTensorWrapper(const TensorInfo& tensor_info,
                                const std::string& tensor_name,
                                QnnTensorWrapper& tensor_wrapper) const;

  // Add to internal tensor wrapper table
  bool AddTensorWrapper(QnnTensorWrapper&& tensor_wrapper);

  // Add to internal param wrapper table
  bool AddParamWrapper(QnnParamWrapper&& param_wrapper);

  const QnnTensorWrapper& GetQnnTensorWrapper(const std::string& tensor_name);

  // Utility function to validate a QNN node. Does not modify this object's state.
  Ort::Status ValidateQnnNode(const std::string& node_name,
                              const std::string& package_name,
                              const std::string& qnn_op_type,
                              std::vector<Qnn_Tensor_t>&& input_tensors,
                              std::vector<Qnn_Tensor_t>&& output_tensors,
                              std::vector<Qnn_Param_t>&& params) const;

  bool CreateQnnNode(const std::string& name,
                     const std::string& package_name,
                     const std::string& type,
                     std::vector<std::string>&& input_names,
                     std::vector<std::string>&& output_names,
                     std::vector<std::string>&& param_tensor_names,
                     bool do_op_validation = false);

  bool ComposeQnnGraph(bool build_json_qnn_graph = false);

  Qnn_GraphHandle_t GetQnnGraph() const { return graph_; }

  std::string GetQnnGraphName() const { return graph_name_; }

  Qnn_ContextHandle_t GetQnnGraphContext() const { return graph_context_; }

  // Move input tensor wrappers to GraphInfo, QnnModelWrapper end of live
  std::vector<QnnTensorWrapper>&& GetGraphInputTensorWrappers() {
    GetGraphInputOutputTensorWrapper(model_input_names_, model_input_tensor_wrappers_);
    return std::move(model_input_tensor_wrappers_);
  }

  // Move output tensor wrappers to GraphInfo, QnnModelWrapper end of live
  std::vector<QnnTensorWrapper>&& GetGraphOutputTensorWrappers() {
    GetGraphInputOutputTensorWrapper(model_output_names_, model_output_tensor_wrappers_);
    return std::move(model_output_tensor_wrappers_);
  }

  Ort::Status GetInitializerTensors(gsl::span<const OrtValueInfo*> initializers) const {
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Graph_GetInitializers(&ort_graph_,
                                                                       initializers.data(),
                                                                       initializers.size()));
    return Ort::Status();
  }

  // Find an initializer by name
  Ort::Status FindInitializer(const std::string& tensor_name,
                              const OrtValueInfo** found_value_info = nullptr) const {
    size_t num_initializers = 0;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Graph_GetNumInitializers(&ort_graph_, &num_initializers));

    std::vector<const OrtValueInfo*> initializers(num_initializers);
    RETURN_IF_ERROR(GetInitializerTensors(initializers));

    for (const OrtValueInfo* value_info : initializers) {
      const char* value_info_name = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoName(value_info, &value_info_name));

      if (std::string(value_info_name) == tensor_name) {
        *found_value_info = value_info;
        return Ort::Status();
      }
    }

    return MAKE_EP_FAIL("Initializer not found");
  }

  const OrtValueInfo* GetConstantTensor(const std::string& tensor_name) const {
    const OrtValueInfo* value_info = nullptr;
    Ort::Status status = FindInitializer(tensor_name, &value_info);
    if (!status.IsOK() || value_info == nullptr) {
      return nullptr;
    }

    bool is_constant_initializer = false;
    OrtStatus* ort_status = api_ptrs_.ort_api.ValueInfo_IsConstantInitializer(value_info, &is_constant_initializer);
    if (ort_status != nullptr) {
      api_ptrs_.ort_api.ReleaseStatus(ort_status);
      return nullptr;
    }

    if (!is_constant_initializer) {
      return nullptr;
    }

    return value_info;
  }

  // This function aims to check if the input to a node is an initializer.
  // Note: the `input` here refers to the input of the node, not the input of the graph.
  bool IsConstantInput(const std::string& input_name) const {
    const OrtValueInfo* value_info = nullptr;
    Ort::Status status = FindInitializer(input_name, &value_info);
    if (!status.IsOK() || value_info == nullptr) {
      return false;
    }

    bool is_constant_initializer = false;
    OrtStatus* ort_status = api_ptrs_.ort_api.ValueInfo_IsConstantInitializer(value_info, &is_constant_initializer);
    if (ort_status != nullptr) {
      api_ptrs_.ort_api.ReleaseStatus(ort_status);
      return false;
    }

    return is_constant_initializer;
  }

  // static bool GetOnnxShape(const NodeArg& node_arg, std::vector<uint32_t>& shape);
  static bool GetOnnxShape(const std::vector<int64_t>& onnx_shape, std::vector<uint32_t>& shape);

  bool IsQnnTensorWrapperExist(const std::string& tensor_name) const;

  bool IsGraphOutput(const std::string& tensor_name) const {
    return output_index_map_.find(tensor_name) != output_index_map_.end();
  }

  bool IsGraphInput(const std::string& tensor_name) const {
    return input_index_map_.find(tensor_name) != input_index_map_.end();
  }

  const nlohmann::json& GetQnnJSONGraph() {
    return json_qnn_graph_.Finalize();
  }

  Qnn_TensorType_t GetTensorType(const std::string& tensor_name) const {
    if (IsConstantInput(tensor_name)) {
      return QNN_TENSOR_TYPE_STATIC;
    } else if (IsGraphInput(tensor_name)) {
      return QNN_TENSOR_TYPE_APP_WRITE;
    } else if (IsGraphOutput(tensor_name)) {
      return QNN_TENSOR_TYPE_APP_READ;
    } else {
      return QNN_TENSOR_TYPE_NATIVE;
    }
  }

  Ort::Status GetTensorInfo(const OrtNodeUnitIODef& tensor, TensorInfo& tensor_info) const;

  Ort::Status AddReshapeNode(const std::string& input_name,
                             const std::string& output_name,
                             const std::vector<uint32_t>& input_shape,
                             const std::vector<uint32_t>& output_shape,
                             const Qnn_DataType_t& tensor_data_type,
                             const QnnQuantParamsWrapper& input_quantize_param,
                             const QnnQuantParamsWrapper& output_quantize_param,
                             bool do_op_validation,
                             bool is_for_input = true,
                             bool is_for_output = false);

  Ort::Status AddReshapeNode(const std::string& input_name,
                             const std::string& output_name,
                             const std::vector<uint32_t>& input_shape,
                             const std::vector<uint32_t>& output_shape,
                             const Qnn_DataType_t& tensor_data_type,
                             const QnnQuantParamsWrapper& quantize_param,
                             bool do_op_validation,
                             bool is_for_input = true,
                             bool is_for_output = false);

  Ort::Status AddTransposeNode(size_t node_index,
                               const std::string& input_name,
                               const std::string& output_name,
                               const std::vector<uint32_t>& input_shape,
                               const std::vector<uint32_t>& transpose_perm,
                               const std::vector<uint32_t>& output_shape,
                               const Qnn_DataType_t& tensor_data_type,
                               const QnnQuantParamsWrapper& quantize_param,
                               bool do_op_validation,
                               bool is_for_input = true,
                               bool is_for_output = false);

  // Transpose NCHW->HWCN for QNN weight
  Ort::Status AddNchwToHwcnTranspose(size_t node_index,
                                     const std::string& input_name,
                                     const std::string& output_name,
                                     const std::vector<uint32_t>& input_shape,
                                     const std::vector<uint32_t>& output_shape,
                                     const Qnn_DataType_t& tensor_data_type,
                                     const QnnQuantParamsWrapper& quantize_param,
                                     bool do_op_validation,
                                     bool is_for_input = true,
                                     bool is_for_output = false,
                                     bool is_3d = false) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Add NCHW->HWCN Transpose node after Conv weight input: " +
                 input_name + " -> " + output_name)
                    .c_str());
    auto perm = is_3d ? nchw2hwcn_perm_3d : nchw2hwcn_perm;
    std::vector<uint32_t> transpose_perm;
    transpose_perm.resize(perm.size());
    std::transform(perm.begin(), perm.end(),
                   transpose_perm.begin(), [](size_t item) -> uint32_t {
                     return gsl::narrow<uint32_t>(item);
                   });
    return AddTransposeNode(node_index, input_name, output_name, input_shape, transpose_perm, output_shape,
                            tensor_data_type, quantize_param, do_op_validation, is_for_input, is_for_output);
  }

  // Tranpose CNHW->HWCN for QNN weight
  Ort::Status AddCnhwToHwcnTranspose(size_t node_index,
                                     const std::string& input_name,
                                     const std::string& output_name,
                                     const std::vector<uint32_t>& input_shape,
                                     const std::vector<uint32_t>& output_shape,
                                     const Qnn_DataType_t& tensor_data_type,
                                     const QnnQuantParamsWrapper& quantize_param,
                                     bool do_op_validation,
                                     bool is_for_input = true,
                                     bool is_for_output = false,
                                     bool is_3d = false) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Add CNHW->HWCN Transpose node after ConvTranspose weight input: " +
                 input_name + " -> " + output_name)
                    .c_str());
    auto perm = is_3d ? cnhw2hwcn_perm_3d : cnhw2hwcn_perm;
    std::vector<uint32_t> transpose_perm;
    transpose_perm.resize(perm.size());
    std::transform(perm.begin(), perm.end(),
                   transpose_perm.begin(), [](size_t item) -> uint32_t {
                     return gsl::narrow<uint32_t>(item);
                   });
    return AddTransposeNode(node_index, input_name, output_name, input_shape, transpose_perm, output_shape,
                            tensor_data_type, quantize_param, do_op_validation, is_for_input, is_for_output);
  }

  Ort::Status UnpackInitializerData(const OrtValueInfo* initializer,
                                    std::vector<uint8_t>& unpacked_tensor) const;

  QnnBackendType GetQnnBackendType() const { return qnn_backend_type_; }

  const OrtGraph& GetOrtGraph() const { return ort_graph_; }

  const OrtApi& GetOrtApi() const { return api_ptrs_.ort_api; }

  // Unpack scales from initializer (1 scale for per-tensor, > 1 for per-axis or per-block).
  // Template parameter T allows handling both float and uint8_t scale types.
  template <typename T = float>
  Ort::Status UnpackScales(const OrtValueInfo* scale_tensor, std::vector<T>& scales) const {
    RETURN_IF(scale_tensor == nullptr, "Given scale(s) to be unpacked is null.");

    const OrtTypeInfo* type_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoTypeInfo(scale_tensor, &type_info));

    const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_type_and_shape_info));
    ONNXTensorElementDataType onnx_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorElementType(tensor_type_and_shape_info, &onnx_data_type));

    // Handle float scales
    if constexpr (std::is_same_v<T, float>) {
      // Verify data type for float scales
      RETURN_IF_NOT(onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                    "Expected scale initializer to be of type FLOAT");

      std::vector<uint8_t> initializer_bytes;
      RETURN_IF_ERROR(UnpackInitializerData(scale_tensor, initializer_bytes));

      gsl::span<const float> src = gsl::make_span(reinterpret_cast<const float*>(initializer_bytes.data()),
                                                  initializer_bytes.size() / sizeof(float));

      scales.insert(scales.end(), src.begin(), src.end());
    }
    // Handle uint8_t scales (for block quantization)
    else if constexpr (std::is_same_v<T, uint8_t>) {
      // Verify data type for uint8_t scales
      RETURN_IF_NOT(onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                    "Expected uint8_t scale initializer to be of type UINT8");

      RETURN_IF_ERROR(UnpackInitializerData(scale_tensor, scales));
    } else {
      return MAKE_EP_FAIL(("Scale ONNX data type `" +
                           std::string(typeid(T).name()) +
                           "` is not supported for unpacking.")
                              .c_str());
    }
    return Ort::Status();
  }

  // Unpack zero-points from initializer and convert to int32_t (1 zero-point for per-tensor, > 1 for per-channel).
  Ort::Status UnpackZeroPoints(const OrtValueInfo* zp_tensor,
                               /*out*/ std::vector<int32_t>& zero_points,
                               /*out*/ ONNXTensorElementDataType& onnx_data_type) const;

  // // Checks if a tensor in the ONNX graph is per-channel quantized.
  Ort::Status IsPerChannelQuantized(const OrtNodeUnitIODef& io_def,
                                    /*out*/ bool& is_per_channel,
                                    /*out*/ int64_t& axis) const;

 private:
  bool CreateQnnInputOutputTensors(const std::string& qnn_node_name,
                                   const std::vector<std::string>& names,
                                   std::vector<Qnn_Tensor_t>& tensor_wrappers,
                                   bool do_op_validation = false);

  bool QnnParamExists(const std::string& param_tensor_name) const;

  bool CreateQnnParamTensors(const std::string& qnn_node_name,
                             const std::vector<std::string>& param_tensor_names,
                             std::vector<Qnn_Param_t>& qnn_params,
                             bool do_op_validation = false);

  bool IsQnnTensorCreated(const std::string& tensor_name) {
    auto pos = tensor_created_map_.find(tensor_name);
    if (pos == tensor_created_map_.end()) {
      return false;
    }
    return pos->second;
  }

  void GetGraphInputOutputTensorWrapper(const std::vector<std::string>& names,
                                        std::vector<QnnTensorWrapper>& wrappers_list);

  const OrtGraph& ort_graph_;
  const Ort::Logger& logger_;
  const QNN_INTERFACE_VER_TYPE& qnn_interface_;
  const Qnn_BackendHandle_t& backend_handle_;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string graph_name_ = "";
  // QNN context that holds the QNN graph referenced by `graph_`
  Qnn_ContextHandle_t graph_context_ = nullptr;

  std::vector<std::string> model_input_names_;
  std::vector<std::string> model_output_names_;
  std::vector<QnnTensorWrapper> model_input_tensor_wrappers_;
  std::vector<QnnTensorWrapper> model_output_tensor_wrappers_;
  // All QnnTensorWrapper for the graph
  std::unordered_map<std::string, QnnTensorWrapper> model_tensors_map_;
  // All QnnParamWrapper for the graph
  std::unordered_map<std::string, QnnParamWrapper> model_params_map_;
  std::vector<QnnOpProperty> qnn_op_property_list_;
  // <tensor_name, qnn_tensor_created> -- true means qnn tensor created in qnn graph
  // it includs normal qnn_tensors and qnn_tensors inside param_tensors
  std::unordered_map<std::string, bool> tensor_created_map_;
  const std::unordered_map<std::string, size_t>& input_index_map_;
  const std::unordered_map<std::string, size_t>& output_index_map_;
  QnnBackendType qnn_backend_type_ = QnnBackendType::CPU;
  ModelSettings model_settings_ = {};
  utils::QnnJSONGraph json_qnn_graph_;
  const ApiPtrs api_ptrs_;
};  // QnnModelWrapper

template <typename T>
inline Ort::Status AddQnnScalar(QnnModelWrapper& qnn_model_wrapper,
                                const size_t& node_index,
                                const std::string& node_name,
                                const T& scalar,
                                const std::string& qnn_scalar_param_name,
                                std::vector<std::string>& param_names) {
  Qnn_Scalar_t qnn_scalar = QNN_SCALAR_INIT;
  if (std::is_same<T, float>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    qnn_scalar.floatValue = static_cast<float>(scalar);
  } else if (std::is_same<T, uint32_t>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
    qnn_scalar.uint32Value = static_cast<uint32_t>(scalar);
  } else if (std::is_same<T, int32_t>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_INT_32;
    qnn_scalar.int32Value = static_cast<int32_t>(scalar);
  } else if (std::is_same<T, int64_t>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_INT_64;
    qnn_scalar.int64Value = static_cast<int64_t>(scalar);
  } else if (std::is_same<T, bool>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_BOOL_8;
    qnn_scalar.bool8Value = static_cast<uint8_t>(scalar);
  } else {
    RETURN_IF(true, "QNN EP: Unsupported scalar dtype");
  }
  QnnParamWrapper qnn_param_wrapper(node_index, node_name, qnn_scalar_param_name, qnn_scalar);
  param_names.push_back(qnn_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(qnn_param_wrapper));
  return Ort::Status();
}

inline Ort::Status AddQnnScalar(QnnModelWrapper& qnn_model_wrapper,
                                const size_t& node_index,
                                const std::string& node_name,
                                const std::string& scalar,
                                const std::string& qnn_scalar_param_name,
                                std::vector<std::string>& param_names) {
  Qnn_Scalar_t qnn_scalar = QNN_SCALAR_INIT;
  qnn_scalar.dataType = QNN_DATATYPE_STRING;
  qnn_scalar.stringValue = scalar.c_str();
  QnnParamWrapper qnn_param_wrapper(node_index, node_name, qnn_scalar_param_name, qnn_scalar);
  param_names.push_back(qnn_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(qnn_param_wrapper));
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
