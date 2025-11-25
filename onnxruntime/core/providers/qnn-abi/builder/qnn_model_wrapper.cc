// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

bool QnnModelWrapper::CreateQnnGraph(const Qnn_ContextHandle_t& context,
                                     const std::string& graph_name,
                                     const QnnGraph_Config_t** graph_configs) {
  if (!graph_name_.empty()) {
    // only one graph is allowed per QnnModel
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Graph" + graph_name + " already initialized.").c_str());
    return false;
  }
  if (context == nullptr) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Invalid Qnn context.");
    return false;
  }
  if (graph_name.length() == 0) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Empty graph name.");
    return false;
  }

  auto rt = qnn_interface_.graphCreate(context, graph_name.c_str(), graph_configs, &graph_);
  if (rt != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
    rt = qnn_interface_.graphRetrieve(context, graph_name.c_str(), &graph_);
    if (rt != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Failed to create Qnn graph: " + graph_name).c_str());
      return false;
    }
  }

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Created Qnn graph: " + graph_name).c_str());

  graph_name_ = graph_name;
  graph_context_ = context;
  return true;
}

bool QnnModelWrapper::IsQnnTensorWrapperExist(const std::string& name) const {
  return model_tensors_map_.find(name) != model_tensors_map_.end();
}

bool QnnModelWrapper::QnnParamExists(const std::string& param_tensor_name) const {
  return model_params_map_.find(param_tensor_name) != model_params_map_.end();
}

Ort::Status QnnModelWrapper::MakeTensorWrapper(const OrtNodeUnitIODef& tensor, QnnTensorWrapper& tensor_wrapper) const {
  const std::string& tensor_name = tensor.name;

  TensorInfo tensor_info = {};
  RETURN_IF_ERROR(GetTensorInfo(tensor, tensor_info));

  std::vector<uint8_t> unpacked_tensor;
  if (tensor_info.is_initializer) {
    RETURN_IF_ERROR(UnpackInitializerData(tensor_info.initializer_tensor, unpacked_tensor));
  }

  Qnn_TensorMemType_t mem_type = QNN_TENSORMEMTYPE_RAW;
  if (true == model_settings_.htp_shared_memory && (IsGraphInput(tensor_name) || IsGraphOutput(tensor_name))) {
    mem_type = QNN_TENSORMEMTYPE_MEMHANDLE;
  }
  tensor_wrapper = QnnTensorWrapper(tensor_name,
                                    GetTensorType(tensor_name),
                                    tensor_info.qnn_data_type,
                                    std::move(tensor_info.quant_param),
                                    std::move(tensor_info.shape),
                                    std::move(unpacked_tensor),
                                    mem_type);
  return Ort::Status();
}

Ort::Status QnnModelWrapper::MakeTensorWrapper(const TensorInfo& tensor_info,
                                               const std::string& tensor_name,
                                               QnnTensorWrapper& tensor_wrapper) const {
  std::vector<uint8_t> unpacked_tensor;
  if (tensor_info.is_initializer) {
    RETURN_IF_ERROR(UnpackInitializerData(tensor_info.initializer_tensor, unpacked_tensor));
  }

  tensor_wrapper = QnnTensorWrapper(tensor_name, GetTensorType(tensor_name), tensor_info.qnn_data_type,
                                    tensor_info.quant_param.Copy(), std::vector<uint32_t>(tensor_info.shape),
                                    std::move(unpacked_tensor));
  return Ort::Status();
}

bool QnnModelWrapper::AddTensorWrapper(QnnTensorWrapper&& tensor_wrapper) {
  // Keep a copy of tensor name sine it will be moved with the wrapper into model_tensors_map_
  std::string tensor_name = tensor_wrapper.GetName();
  if (tensor_name.length() == 0) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Invalid tensor encountered empty name.");
    return false;
  }

  if (IsQnnTensorWrapperExist(tensor_name) == true) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor exist already: " + tensor_name).c_str());
    return true;
  }

  const Qnn_TensorType_t& qnn_tensor_type = tensor_wrapper.GetTensorType();
  // save created tensors for later lookup to populate graph node construction
  model_tensors_map_.emplace(tensor_name, std::move(tensor_wrapper));

  // save network input/outputs tensors to use for setting the Qnn graph's
  // input and output tensors for populating GraphInfo for caller
  if (qnn_tensor_type == QNN_TENSOR_TYPE_APP_WRITE) {
    model_input_names_.push_back(tensor_name);
  } else if (qnn_tensor_type == QNN_TENSOR_TYPE_APP_READ) {
    model_output_names_.push_back(tensor_name);
  }

  return true;
}

bool QnnModelWrapper::AddParamWrapper(QnnParamWrapper&& param_wrapper) {
  // Keep a copy of tensor name since it will be moved with the wrapper into model_params_map_
  std::string param_tensor_name = param_wrapper.GetParamTensorName();
  if (param_tensor_name.length() == 0) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Invalid parameter encountered empty name.");
    return false;
  }

  if (QnnParamExists(param_tensor_name) == true) {
    return true;
  }

  // save created tensors for later lookup to populate graph node construction
  model_params_map_.emplace(param_tensor_name, std::move(param_wrapper));

  return true;
}

const QnnTensorWrapper& QnnModelWrapper::GetQnnTensorWrapper(const std::string& tensor_name) {
  auto map_iter = model_tensors_map_.find(tensor_name);
  if (map_iter != model_tensors_map_.end()) {
    return (map_iter->second);
  }

  ORT_CXX_API_THROW("Qnn tensor not exist: " + tensor_name, ORT_EP_FAIL);
}

bool QnnModelWrapper::CreateQnnInputOutputTensors(const std::string& qnn_node_name,
                                                  const std::vector<std::string>& tensor_names,
                                                  std::vector<Qnn_Tensor_t>& qnn_tensors,
                                                  bool do_op_validation) {
  for (const auto& tensor_name : tensor_names) {
    auto it = model_tensors_map_.find(tensor_name);
    if (it == model_tensors_map_.end()) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Input name not exist: " + tensor_name).c_str());
      return false;
    }

    // During graph partitioning, we only need to do op validation, it's not required to create Qnn graph tensor
    // We only need to create the Qnn graph tensor during Compile to create Qnn graph
    if (!do_op_validation) {
      std::string error_string;
      auto rt = it->second.CreateQnnGraphTensor(qnn_interface_, graph_, qnn_node_name, tensor_created_map_, error_string);
      if (!rt) {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, error_string.c_str());
        return false;
      }
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor: " + tensor_name + " created. " + error_string).c_str());
    }

    qnn_tensors.push_back(it->second.GetQnnTensor());
  }
  return true;
}

bool QnnModelWrapper::CreateQnnParamTensors(const std::string& qnn_node_name,
                                            const std::vector<std::string>& param_tensor_names,
                                            std::vector<Qnn_Param_t>& qnn_params,
                                            bool do_op_validation) {
  for (const auto& param_tensor_name : param_tensor_names) {
    auto it = model_params_map_.find(param_tensor_name);
    if (it == model_params_map_.end()) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Parameter name not exist: " + param_tensor_name).c_str());
      return false;
    }

    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Add parameter tensor: " + it->second.GetName()).c_str());
    if (!do_op_validation) {
      std::string error_string;
      auto rt = it->second.CreateQnnGraphParam(qnn_interface_, graph_, qnn_node_name, tensor_created_map_, error_string);
      if (!rt) {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, error_string.c_str());
        return false;
      }
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor: " + param_tensor_name + " created. " + error_string).c_str());
    }

    qnn_params.push_back(it->second.GetQnnParam());
  }

  return true;
}

Ort::Status QnnModelWrapper::ValidateQnnNode(const std::string& node_name,
                                             const std::string& package_name,
                                             const std::string& qnn_op_type,
                                             std::vector<Qnn_Tensor_t>&& input_tensors,
                                             std::vector<Qnn_Tensor_t>&& output_tensors,
                                             std::vector<Qnn_Param_t>&& params) const {
  QnnOpConfigWrapper op_config_wrapper(node_name,
                                       package_name,
                                       qnn_op_type,
                                       std::move(input_tensors),
                                       std::move(output_tensors),
                                       std::move(params));

  std::string error_msg;
  RETURN_IF_NOT(op_config_wrapper.QnnGraphOpValidation(qnn_interface_, backend_handle_, error_msg), error_msg.c_str());

  return Ort::Status();
}

bool QnnModelWrapper::CreateQnnNode(const std::string& qnn_node_name,
                                    const std::string& package_name,
                                    const std::string& qnn_node_type,
                                    std::vector<std::string>&& input_names,
                                    std::vector<std::string>&& output_names,
                                    std::vector<std::string>&& param_tensor_names,
                                    bool do_op_validation) {
  if (do_op_validation) {
    std::vector<Qnn_Tensor_t> input_tensors;
    std::vector<Qnn_Tensor_t> output_tensors;
    std::vector<Qnn_Param_t> params;
    if (!CreateQnnInputOutputTensors(qnn_node_name, input_names, input_tensors, do_op_validation)) {
      return false;
    }

    if (!CreateQnnInputOutputTensors(qnn_node_name, output_names, output_tensors, do_op_validation)) {
      return false;
    }

    if (!CreateQnnParamTensors(qnn_node_name, param_tensor_names, params, do_op_validation)) {
      return false;
    }

    QnnOpConfigWrapper op_config_wrapper(qnn_node_name,
                                         package_name,
                                         qnn_node_type,
                                         std::move(input_tensors),
                                         std::move(output_tensors),
                                         std::move(params));

    using namespace onnxruntime::qnn::utils;

    std::ostringstream oss;
    oss << op_config_wrapper;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

    std::string error_msg;
    bool rt = op_config_wrapper.QnnGraphOpValidation(qnn_interface_, backend_handle_, error_msg);
    if (!rt) {
      // TODO(adrianlizarraga): Return a Status with the error message so that aggregated logs show a more
      // specific validation error (instead of "failed to add node").
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING, error_msg.c_str());
    }
    return rt;
  } else {
    QnnOpProperty qnn_op(qnn_node_name, package_name, qnn_node_type,
                         std::move(input_names), std::move(output_names), std::move(param_tensor_names));
    qnn_op_property_list_.push_back(std::move(qnn_op));
    return true;
  }
}

bool QnnModelWrapper::ComposeQnnGraph(bool build_json_qnn_graph) {
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Compose Qnn Graph.");
  if (qnn_op_property_list_.empty()) {
    return false;
  }

  for (const auto& op_property : qnn_op_property_list_) {
    std::vector<Qnn_Tensor_t> input_tensors;
    std::vector<Qnn_Tensor_t> output_tensors;
    std::vector<Qnn_Param_t> params;
    if (!CreateQnnInputOutputTensors(op_property.GetNodeName(), op_property.GetInputNames(), input_tensors)) {
      return false;
    }

    if (!CreateQnnInputOutputTensors(op_property.GetNodeName(), op_property.GetOutputNames(), output_tensors)) {
      return false;
    }

    if (!CreateQnnParamTensors(op_property.GetNodeName(), op_property.GetParamTensorNames(), params)) {
      return false;
    }

    QnnOpConfigWrapper op_config_wrapper(op_property.GetNodeName(),
                                         op_property.GetPackageName(),
                                         op_property.GetNodeType(),
                                         std::move(input_tensors),
                                         std::move(output_tensors),
                                         std::move(params));

    using namespace onnxruntime::qnn::utils;

    std::ostringstream oss;
    oss << op_config_wrapper;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

    std::string error_msg;
    bool rt = op_config_wrapper.CreateQnnGraphOp(qnn_interface_, graph_, error_msg);
    if (!rt) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, error_msg.c_str());
      return false;
    }

    if (build_json_qnn_graph) {
      json_qnn_graph_.AddOp(op_config_wrapper);
    }
  }

  return true;
}

bool QnnModelWrapper::GetOnnxShape(const std::vector<int64_t>& onnx_shape, std::vector<uint32_t>& shape) {
  // Set shape to 1 for scalar.
  if (onnx_shape.size() < 1) {
    shape.push_back(1);
    return true;
  }

  for (const int64_t& dim : onnx_shape) {
    if (dim < 0) {
      return false;
    }
    shape.push_back(SafeInt<uint32_t>(dim));
  }

  return true;
}

Ort::Status QnnModelWrapper::UnpackZeroPoints(const OrtValueInfo* zp_tensor,
                                              /*out*/ std::vector<int32_t>& zero_points,
                                              /*out*/ ONNXTensorElementDataType& onnx_data_type) const {
  RETURN_IF(zp_tensor == nullptr, "Given zero point(s) to be unpacked is null.");

  const OrtTypeInfo* type_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoTypeInfo(zp_tensor, &type_info));

  const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_type_and_shape_info));
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorElementType(tensor_type_and_shape_info, &onnx_data_type));

  std::vector<uint8_t> initializer_bytes;

  RETURN_IF_ERROR(UnpackInitializerData(zp_tensor, initializer_bytes));

  // Helper to transform zero points (QNN uses -offset for some reason)
  auto transform_zero_points = [&zero_points](auto input_span) {
    std::transform(input_span.begin(), input_span.end(), std::back_inserter(zero_points),
                   [](auto zp) -> int32_t {
                     return -static_cast<int32_t>(zp);
                   });
  };

  switch (onnx_data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4: {  // INT4 zero-points are unpacked as 8-bit values for QNN
      auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(initializer_bytes));
      std::transform(int8_span.begin(), int8_span.end(), std::back_inserter(zero_points),
                     [](int8_t masked_zp) -> int32_t {
                       // We currently unpack int4 as int8 but with the top 4-bits masked off due to QNN bug.
                       // Need to undo the masking so that the zero-point value is correct.
                       // (Not really a problem yet because QNN only supports symmetric INT4 quantization with zp == 0).
                       int8_t zp = Int4x2::SignExtendLower4Bits(std::byte(masked_zp));
                       return -static_cast<int32_t>(zp);
                     });
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      transform_zero_points(ReinterpretAsSpan<const int8_t>(gsl::make_span(initializer_bytes)));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:  // UINT4 zero-points are unpacked as 8-bit values for QNN
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      transform_zero_points(ReinterpretAsSpan<const uint8_t>(gsl::make_span(initializer_bytes)));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      transform_zero_points(ReinterpretAsSpan<const uint16_t>(gsl::make_span(initializer_bytes)));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      transform_zero_points(ReinterpretAsSpan<const int16_t>(gsl::make_span(initializer_bytes)));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      transform_zero_points(ReinterpretAsSpan<const int32_t>(gsl::make_span(initializer_bytes)));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      transform_zero_points(ReinterpretAsSpan<const uint32_t>(gsl::make_span(initializer_bytes)));
      break;
    default:
      return MAKE_EP_FAIL(("Zero-point ONNX data type `" + std::to_string(onnx_data_type) +
                           "` is not supported.")
                              .c_str());
  }

  return Ort::Status();
}

// Checks if a tensor in the ONNX graph is per-channel quantized.
Ort::Status QnnModelWrapper::IsPerChannelQuantized(const OrtNodeUnitIODef& io_def,
                                                   /*out*/ bool& is_per_channel,
                                                   /*out*/ int64_t& axis) const {
  if (!io_def.quant_param) {
    is_per_channel = false;
    return Ort::Status();
  }

  const OrtValueInfo* scale_tensor = io_def.quant_param->scale;
  RETURN_IF(scale_tensor == nullptr, "Given IO def has null scale.");
  std::vector<int64_t> scale_shape = utils::GetInitializerShape(scale_tensor, api_ptrs_.ort_api);

  // Check the number of scale values to determine if the tensor is per-channel.
  // This is consistent with CPU EP's Quant/Dequant logic. We can't use the presence of an axis because even a
  // per-channel DQ/Q op may not have an explicit axis attribute (assumed to default to 1 if missing).
  int64_t total_number_of_elements = std::accumulate(scale_shape.begin(),
                                                     scale_shape.end(),
                                                     static_cast<int64_t>(1),
                                                     std::multiplies<int64_t>());
  const bool is_scalar_or_1_elem_vector = scale_shape.size() == 0 ||
                                          (scale_shape.size() == 1 && total_number_of_elements == 1);

  is_per_channel = !is_scalar_or_1_elem_vector;

  if (is_per_channel) {
    axis = io_def.quant_param->axis.value_or(1);  // 1 is default axis for Q/DQ ops.
    if (axis < 0) {
      // Normalize negative axis by adding rank.
      std::vector<int64_t> tensor_shape = io_def.shape;
      RETURN_IF_NOT(!tensor_shape.empty(), "NULL tensor shape proto");

      const auto rank = tensor_shape.size();
      RETURN_IF_NOT(rank > 0, "Per-channel quantized tensor should be of rank > 0");

      axis += rank;
    }
  }

  return Ort::Status();
}

Ort::Status QnnModelWrapper::GetTensorInfo(const OrtNodeUnitIODef& tensor, TensorInfo& tensor_info) const {
  const std::string& name = tensor.name;

  // Fill in quantization param info.
  RETURN_IF_ERROR(tensor_info.quant_param.Init(*this, tensor));

  // Fill in QNN data type.
  tensor_info.qnn_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(tensor.quant_param.has_value(),
                                        tensor.type,
                                        tensor_info.qnn_data_type));

  // Fill in shape.
  RETURN_IF_NOT(GetOnnxShape(tensor.shape, tensor_info.shape), "Cannot get shape");

  // Fill in initializer info.
  tensor_info.is_initializer = IsConstantInput(name);
  if (tensor_info.is_initializer) {
    tensor_info.initializer_tensor = GetConstantTensor(name);
  }

  return Ort::Status();
}

Ort::Status QnnModelWrapper::AddReshapeNode(const std::string& input_name, const std::string& output_name,
                                            const std::vector<uint32_t>& input_shape,
                                            const std::vector<uint32_t>& output_shape,
                                            const Qnn_DataType_t& tensor_data_type,
                                            const QnnQuantParamsWrapper& input_quantize_param,
                                            const QnnQuantParamsWrapper& output_quantize_param, bool do_op_validation,
                                            bool is_for_input, bool is_for_output) {
  QnnTensorWrapper input_tensorwrapper(input_name, is_for_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_NATIVE,
                                       tensor_data_type, input_quantize_param.Copy(),
                                       std::vector<uint32_t>(input_shape));
  RETURN_IF_NOT(AddTensorWrapper(std::move(input_tensorwrapper)),
                "QNN EP: Failed to add input tensor for inserted Reshape.");

  Qnn_TensorType_t tensor_type = is_for_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, tensor_data_type, output_quantize_param.Copy(),
                                        std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(AddTensorWrapper(std::move(output_tensorwrapper)),
                "QNN EP: Failed to add output tensor for inserted Reshape.");

  RETURN_IF_NOT(CreateQnnNode(utils::GetUniqueName(output_name, QNN_OP_RESHAPE),
                              QNN_OP_PACKAGE_NAME_QTI_AISW,
                              QNN_OP_RESHAPE,
                              {input_name},
                              {output_name},
                              {},
                              do_op_validation),
                "QNN EP: Failed to create manually inserted Qnn Reshape node.");

  return Ort::Status();
}

Ort::Status QnnModelWrapper::AddReshapeNode(const std::string& input_name, const std::string& output_name,
                                            const std::vector<uint32_t>& input_shape,
                                            const std::vector<uint32_t>& output_shape,
                                            const Qnn_DataType_t& tensor_data_type,
                                            const QnnQuantParamsWrapper& quantize_param, bool do_op_validation,
                                            bool is_for_input, bool is_for_output) {
  // Do not allow QNN EP to insert Reshape nodes with per-channel quantization on dynamic tensors
  // if only one quantization param is provided.
  RETURN_IF(quantize_param.IsPerChannel(), "Do not support inserted Reshape nodes with per-channel quantization");
  return AddReshapeNode(input_name, output_name, input_shape, output_shape, tensor_data_type, quantize_param,
                        quantize_param, do_op_validation, is_for_input, is_for_output);
}

Ort::Status QnnModelWrapper::AddTransposeNode(size_t node_index,
                                              const std::string& input_name,
                                              const std::string& output_name,
                                              const std::vector<uint32_t>& input_shape,
                                              const std::vector<uint32_t>& transpose_perm,
                                              const std::vector<uint32_t>& output_shape,
                                              const Qnn_DataType_t& tensor_data_type,
                                              const QnnQuantParamsWrapper& quantize_param,
                                              bool do_op_validation,
                                              bool is_for_input,
                                              bool is_for_output) {
  // Do not allow QNN EP to insert transpose nodes with per-channel quantization on dynamic tensors.
  // We could technically support this by transposing the quantization param's axis value, but
  // we don't need this right now.
  RETURN_IF(quantize_param.IsPerChannel(),
            "Do not support inserted Transpose nodes with per-channel quantization");
  // No need to add this for output nodes as it is added as output tensor for previous node
  if (is_for_input) {
    Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    QnnTensorWrapper input_tensorwrapper(input_name,
                                         tensor_type,
                                         tensor_data_type,
                                         quantize_param.Copy(),
                                         std::vector<uint32_t>(input_shape));
    RETURN_IF_NOT(AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  uint32_t perm_size = static_cast<uint32_t>(transpose_perm.size());
  std::vector<uint32_t> perm_dim{perm_size};
  std::vector<uint32_t> transpose_perm_copy = transpose_perm;
  QnnParamWrapper transpose_param(node_index, output_name, QNN_OP_TRANSPOSE_PARAM_PERM, std::move(perm_dim), std::move(transpose_perm_copy));
  std::string param_tensor_name(transpose_param.GetParamTensorName());
  RETURN_IF_NOT(AddParamWrapper(std::move(transpose_param)), "Failed to add tensor.");
  Qnn_TensorType_t tensor_type = (false == is_for_output) ? QNN_TENSOR_TYPE_NATIVE : QNN_TENSOR_TYPE_APP_READ;
  std::vector<uint32_t> output_shape_copy = output_shape;
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        tensor_data_type,
                                        quantize_param.Copy(),
                                        std::move(output_shape_copy));
  RETURN_IF_NOT(AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  RETURN_IF_NOT(CreateQnnNode(utils::GetUniqueName(output_name, QNN_OP_TRANSPOSE),
                              QNN_OP_PACKAGE_NAME_QTI_AISW,
                              QNN_OP_TRANSPOSE,
                              {input_name},
                              {output_name},
                              {param_tensor_name},
                              do_op_validation),
                "QNN EP: Failed to create manually inserted Qnn Transpose node.");

  return Ort::Status();
}

void QnnModelWrapper::GetGraphInputOutputTensorWrapper(const std::vector<std::string>& tensor_name_list,
                                                       std::vector<QnnTensorWrapper>& wrappers_list) {
  for (const auto& tensor_name : tensor_name_list) {
    auto it = model_tensors_map_.find(tensor_name);
    if (it == model_tensors_map_.end()) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Model input or output name not exist: " + tensor_name + ". Could cause execution error.").c_str());
      break;
    }
    // It's safe to move QnnTensorWrapper out of model_tensors_map_
    // since this call happens when QnnModelWrapper end of live
    wrappers_list.push_back(std::move(it->second));
    model_tensors_map_.erase(tensor_name);
  }

  return;
}

Ort::Status QnnModelWrapper::UnpackInitializerData(const OrtValueInfo* initializer,
                                                   std::vector<uint8_t>& unpacked_tensor) const {
  const ORTCHAR_T* model_path = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Graph_GetModelPath(&ort_graph_, &model_path));
  RETURN_IF_ERROR(utils::UnpackInitializerData(api_ptrs_.ort_api,
                                               initializer,
                                               std::filesystem::path(model_path),
                                               unpacked_tensor));

  const OrtTypeInfo* type_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoTypeInfo(initializer, &type_info));
  const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_type_and_shape_info));
  ONNXTensorElementDataType onnx_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorElementType(tensor_type_and_shape_info, &onnx_data_type));

  // If this is an int4, we need to unpack it because QNN treats int4 as a full int8.
  if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) {
    std::vector<int64_t> shape = utils::GetInitializerShape(initializer, api_ptrs_.ort_api);
    const size_t num_int4_elems = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    RETURN_IF_ERROR(utils::UnpackInt4ToInt8<true>(num_int4_elems, unpacked_tensor));
  } else if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4) {
    std::vector<int64_t> shape = utils::GetInitializerShape(initializer, api_ptrs_.ort_api);
    const size_t num_uint4_elems = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    RETURN_IF_ERROR(utils::UnpackInt4ToInt8<false>(num_uint4_elems, unpacked_tensor));
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
