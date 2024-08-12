// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "qnn_model_wrapper.h"
//#include "core/common/safeint.h"
//#include "core/framework/tensorprotoutils.h"
//#include "core/providers/shared/utils/utils.h"
#include "qnn_utils.h"
#include "core/framework/int4.h"
#include "onnx/onnx_pb.h"

namespace onnxruntime {
namespace qnn {

bool QnnModelWrapper::CreateQnnGraph(const Qnn_ContextHandle_t& context,
                                     const std::string& graph_name,
                                     const QnnGraph_Config_t** graph_configs) {
  if (!graph_name_.empty()) {
    // only one graph is allowed per QnnModel
//    LOGS(logger_, ERROR) << "Graph " << graph_name << " already initialized.";
    return false;
  }
  if (context == nullptr) {
//    LOGS(logger_, ERROR) << "Invalid Qnn context.";
    return false;
  }
  if (graph_name.length() == 0) {
//    LOGS(logger_, ERROR) << "Empty grpah name.";
    return false;
  }

  graph_name_ = graph_name;
  auto rt = qnn_interface_.graphCreate(context, graph_name_.c_str(), graph_configs, &graph_);
  if (rt != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
    rt = qnn_interface_.graphRetrieve(context, graph_name_.c_str(), &graph_);
    if (rt != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
//      LOGS(logger_, ERROR) << "Failed to create Qnn graph: " << graph_name;
      return false;
    }
  }
//  LOGS(logger_, VERBOSE) << "Created Qnn graph: " << graph_name;

  return true;
}

bool QnnModelWrapper::IsQnnTensorWrapperExist(const std::string& name) const {
  return model_tensors_map_.find(name) != model_tensors_map_.end();
}

bool QnnModelWrapper::IsQnnParamExit(const std::string& param_tensor_name) const {
  return model_params_map_.find(param_tensor_name) != model_params_map_.end();
}

//Status QnnModelWrapper::MakeTensorWrapper(const NodeUnitIODef& tensor, QnnTensorWrapper& tensor_wrapper) const {
//  const std::string& tensor_name = tensor.node_arg.Name();
//
//  TensorInfo tensor_info = {};
//  ORT_RETURN_IF_ERROR(GetTensorInfo(tensor, tensor_info));
//
//  std::vector<uint8_t> unpacked_tensor;
//  if (tensor_info.is_initializer) {
//    ORT_RETURN_IF_ERROR(UnpackInitializerData(*tensor_info.initializer_tensor, unpacked_tensor));
//  }
//
//  tensor_wrapper = QnnTensorWrapper(tensor_name, GetTensorType(tensor_name), tensor_info.qnn_data_type,
//                                    std::move(tensor_info.quant_param), std::move(tensor_info.shape),
//                                    std::move(unpacked_tensor));
//  return Status::OK();
//}

bool QnnModelWrapper::AddTensorWrapper(QnnTensorWrapper&& tensor_wrapper) {
  // Keep a copy of tensor name sine it will be moved with the wrapper into model_tensors_map_
  std::string tensor_name = tensor_wrapper.GetName();
  if (tensor_name.length() == 0) {
//    LOGS(logger_, ERROR) << "Invalid tensor encountered empty name.";
    return false;
  }

  if (IsQnnTensorWrapperExist(tensor_name) == true) {
//    LOGS(logger_, VERBOSE) << "Tensor exist already: " << tensor_name;
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
  // Keep a copy of tensor name sine it will be moved with the wrapper into model_params_map_
  std::string param_tensor_name = param_wrapper.GetParamTensorName();
  if (param_tensor_name.length() == 0) {
//    LOGS(logger_, ERROR) << "Invalid parameter encountered empty name.";
    return false;
  }

  if (IsQnnParamExit(param_tensor_name) == true) {
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

  ORT_THROW("Qnn tensor not exist: ", tensor_name);
}

bool QnnModelWrapper::CreateQnnInputOutputTensors(const std::string& qnn_node_name,
                                                  const std::vector<std::string>& tensor_names,
                                                  std::vector<Qnn_Tensor_t>& qnn_tensors,
                                                  bool do_op_validation) {
  for (const auto& tensor_name : tensor_names) {
    auto it = model_tensors_map_.find(tensor_name);
    if (it == model_tensors_map_.end()) {
//      LOGS(logger_, ERROR) << "Input name not exist: " << tensor_name;
      return false;
    }

    // During graph patitioning, we only need to do op validation, it's not required to create Qnn graph tensor
    // We only need to creat the Qnn graph tensor during Compile to create Qnn graph
    if (!do_op_validation) {
      std::string error_string;
      auto rt = it->second.CreateQnnGraphTensor(qnn_interface_, graph_, qnn_node_name, tensor_created_map_, error_string);
      if (!rt) {
//        LOGS(logger_, ERROR) << error_string;
        return false;
      }
//      LOGS(logger_, VERBOSE) << "Tensor: " << tensor_name << " created. " << error_string;
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
//      LOGS(logger_, ERROR) << "Parameter name not exist: " << param_tensor_name;
      return false;
    }

//    LOGS(logger_, VERBOSE) << "Add parameter tensor: " << it->second.GetName();
    if (!do_op_validation) {
      std::string error_string;
      auto rt = it->second.CreateQnnGraphParam(qnn_interface_, graph_, qnn_node_name, tensor_created_map_, error_string);
      if (!rt) {
//        LOGS(logger_, ERROR) << error_string;
        return false;
      }
//      LOGS(logger_, VERBOSE) << "Tensor: " << param_tensor_name << " created. " << error_string;
    }

    qnn_params.push_back(it->second.GetQnnParam());
  }

  return true;
}

Status QnnModelWrapper::ValidateQnnNode(const std::string& node_name,
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
  ORT_RETURN_IF_NOT(op_config_wrapper.QnnGraphOpValidation(qnn_interface_, backend_handle_, error_msg), error_msg);

  return Status::OK();
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
//    LOGS(logger_, VERBOSE) << op_config_wrapper;

    std::string error_msg;
    bool rt = op_config_wrapper.QnnGraphOpValidation(qnn_interface_, backend_handle_, error_msg);
    if (!rt) {
//      LOGS(logger_, WARNING) << error_msg;
    }
    return rt;
  } else {
    QnnOpProperty qnn_op(qnn_node_name, package_name, qnn_node_type,
                         std::move(input_names), std::move(output_names), std::move(param_tensor_names));
    qnn_op_property_list_.push_back(std::move(qnn_op));
    return true;
  }
}

bool QnnModelWrapper::ComposeQnnGraph() {
//  LOGS(logger_, VERBOSE) << "Compose Qnn Graph.";
  // ORT_RETURN_IF(qnn_op_property_list_.empty(), "Empty Qnn op list, no graph to compose.");
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
//    LOGS(logger_, VERBOSE) << op_config_wrapper;

    std::string error_msg;
    bool rt = op_config_wrapper.CreateQnnGraphOp(qnn_interface_, graph_, error_msg);
    if (!rt) {
//      LOGS(logger_, ERROR) << error_msg;
      return false;
    }
  }

  return true;
}

//bool QnnModelWrapper::GetOnnxShape(const NodeArg& node_arg, std::vector<uint32_t>& shape) {
//  const auto* shape_proto = node_arg.Shape();
//  if (shape_proto == nullptr) {
//    return false;
//  }
//
//  // For Scalar data, we need to set shape to 1 for QNN
//  if (shape_proto->dim_size() < 1) {
//    shape.push_back(1);
//    return true;
//  }
//
//  // We already checked the shape has no dynamic dimension
//  for (const auto& dim : shape_proto->dim()) {
////    shape.push_back(SafeInt<uint32_t>(dim.dim_value()));
//  }
//
//  return true;
//}

Status QnnModelWrapper::UnpackZeroPoints(const std::string& initializer_name,
                                         /*out*/ std::vector<int32_t>& zero_points,
                                         /*out*/ int32_t& onnx_data_type) const {
//  const auto& graph_initializers = GetInitializerTensors();
//  auto iter = graph_initializers.find(initializer_name);
//  ORT_RETURN_IF(iter == graph_initializers.end(), "Unable to find initializer for zero-point(s): ",
//                initializer_name.c_str());
//  gsl::not_null<const onnx::TensorProto*> zp_tensor_proto = iter->second;
//
//  ORT_RETURN_IF_NOT(zp_tensor_proto->has_data_type(), "Expected zero-point initializer ", initializer_name.c_str(),
//                    " to have a proto data type.");
//
//  onnx_data_type = zp_tensor_proto->data_type();
//  std::vector<uint8_t> initializer_bytes;
//
//  ORT_RETURN_IF_ERROR(UnpackInitializerData(*zp_tensor_proto, initializer_bytes));
//
//  switch (onnx_data_type) {
//    // QNN use -offset for some reason
//    case ONNX_NAMESPACE::TensorProto_DataType_INT4:  // INT4 zero-points are unpacked as 8-bit values for QNN
//    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
//      auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(initializer_bytes));
//      std::transform(int8_span.begin(), int8_span.end(), std::back_inserter(zero_points),
//                     [](int8_t zp) -> int32_t {
//                       return -static_cast<int32_t>(zp);
//                     });
//      break;
//    }
//    case ONNX_NAMESPACE::TensorProto_DataType_UINT4:  // UINT4 zero-points are unpacked as 8-bit values for QNN
//    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
//      auto uint8_span = ReinterpretAsSpan<const uint8_t>(gsl::make_span(initializer_bytes));
//      std::transform(uint8_span.begin(), uint8_span.end(), std::back_inserter(zero_points),
//                     [](uint8_t zp) -> int32_t {
//                       return -static_cast<int32_t>(zp);
//                     });
//      break;
//    }
//    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
//      auto uint16_span = ReinterpretAsSpan<const uint16_t>(gsl::make_span(initializer_bytes));
//      std::transform(uint16_span.begin(), uint16_span.end(), std::back_inserter(zero_points),
//                     [](uint16_t zp) -> int32_t {
//                       return -static_cast<int32_t>(zp);
//                     });
//      break;
//    }
//    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
//      auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(initializer_bytes));
//      std::transform(int16_span.begin(), int16_span.end(), std::back_inserter(zero_points),
//                     [](int16_t zp) -> int32_t {
//                       return -static_cast<int32_t>(zp);
//                     });
//      break;
//    }
//    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
//      auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(initializer_bytes));
//      std::transform(int32_span.begin(), int32_span.end(), std::back_inserter(zero_points),
//                     [](int32_t zp) -> int32_t {
//                       return -zp;
//                     });
//      break;
//    }
//    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
//      auto uint32_span = ReinterpretAsSpan<const uint32_t>(gsl::make_span(initializer_bytes));
//      std::transform(uint32_span.begin(), uint32_span.end(), std::back_inserter(zero_points),
//                     [](uint32_t zp) -> int32_t {
//                       return -static_cast<int32_t>(zp);
//                     });
//      break;
//    }
//    default: {
//      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Zero-point ONNX data type `", onnx_data_type,
//                             "` is not supported.");
//    }
//  }

  return Status::OK();
}

Status QnnModelWrapper::UnpackScales(const std::string& initializer_name, std::vector<float>& scales) const {
//  const auto& graph_initializers = GetInitializerTensors();
//  auto iter = graph_initializers.find(initializer_name);
//  ORT_RETURN_IF(iter == graph_initializers.end(), "Unable to find initializer for scale(s): ",
//                initializer_name.c_str());
//  gsl::not_null<const onnx::TensorProto*> scale_tensor_proto = iter->second;
//
//  ORT_RETURN_IF_NOT(scale_tensor_proto->has_data_type(), "Expected scale initializer ", initializer_name.c_str(),
//                    " to have a proto data type.");
//  ORT_RETURN_IF_NOT(scale_tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
//                    "Expected scale initializer to be of type FLOAT");
//
//  std::vector<uint8_t> initializer_bytes;
//
//  ORT_RETURN_IF_ERROR(UnpackInitializerData(*scale_tensor_proto, initializer_bytes));
//
//  gsl::span<const float> src = gsl::make_span(reinterpret_cast<const float*>(initializer_bytes.data()),
//                                              initializer_bytes.size() / sizeof(float));
//
//  scales.insert(scales.end(), src.begin(), src.end());
  return Status::OK();
}

// Checks if a tensor in the ONNX graph is per-channel quantized.
//Status QnnModelWrapper::IsPerChannelQuantized(const onnxruntime::NodeUnitIODef& io_def,
//                                              /*out*/ bool& is_per_channel,
//                                              /*out*/ int64_t& axis) const {
//  if (!io_def.quant_param) {
//    is_per_channel = false;
//    return Status::OK();
//  }
//
//  const std::string& scale_name = io_def.quant_param->scale.Name();
//  const auto& graph_initializers = GetInitializerTensors();
//  auto iter = graph_initializers.find(scale_name);
//  ORT_RETURN_IF(iter == graph_initializers.end(), "Unable to find initializer for scale(s): ",
//                scale_name.c_str());
//  gsl::not_null<const onnx::TensorProto*> scale_tensor_proto = iter->second;
//  TensorShape scale_shape = onnxruntime::utils::GetTensorShapeFromTensorProto(*scale_tensor_proto);
//
//  // Check the number of scale values to determine if the tensor is per-channel.
//  // This is consistent with CPU EP's Quant/Dequant logic. We can't use the presence of an axis because even a
//  // per-channel DQ/Q op may not have an explicit axis attribute (assumed to default to 1 if missing).
//  const bool is_scalar_or_1_elem_vector = scale_shape.NumDimensions() == 0 ||
//                                          (scale_shape.NumDimensions() == 1 && scale_shape.Size() == 1);
//
//  is_per_channel = !is_scalar_or_1_elem_vector;
//
//  if (is_per_channel) {
//    axis = io_def.quant_param->axis.value_or(1);  // 1 is default axis for Q/DQ ops.
//  }
//
//  return Status::OK();
//}

//Status QnnModelWrapper::GetTensorInfo(const NodeUnitIODef& input, TensorInfo& tensor_info) const {
//  const std::string& name = input.node_arg.Name();
//
//  // Fill in quantization param info.
//  ORT_RETURN_IF_ERROR(tensor_info.quant_param.Init(*this, input));
//
//  // Fill in QNN data type.
//  tensor_info.qnn_data_type = QNN_DATATYPE_FLOAT_32;
//  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(input.quant_param.has_value(), input.node_arg.TypeAsProto(),
//                                            tensor_info.qnn_data_type));
//
//  // Fill in shape.
//  ORT_RETURN_IF_NOT(GetOnnxShape(input.node_arg, tensor_info.shape), "Cannot get shape");
//
//  // Fill in initializer info.
//  tensor_info.is_initializer = IsInitializerInput(name);
//  if (tensor_info.is_initializer) {
//    tensor_info.initializer_tensor = GetInitializerTensors().at(name);
//  }
//
//  return Status::OK();
//}

Status QnnModelWrapper::AddReshapeNode(const std::string& input_name,
                                       const std::string& output_name,
                                       const std::vector<uint32_t>& input_shape,
                                       const std::vector<uint32_t>& output_shape,
                                       const Qnn_DataType_t& tensor_data_type,
                                       const QnnQuantParamsWrapper& quantize_param,
                                       bool do_op_validation,
                                       bool is_for_input,
                                       bool is_for_output) {
  // Do not allow QNN EP to insert Reshape nodes with per-channel quantization on dynamic tensors.
  // We could technically support this by shifting the quantization param's axis value, but
  // we don't need this right now.
  ORT_RETURN_IF(quantize_param.IsPerChannel(),
                "Do not support inserted Reshape nodes with per-channel quantization");
  QnnTensorWrapper input_tensorwrapper(input_name,
                                       is_for_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_NATIVE,
                                       tensor_data_type,
                                       quantize_param.Copy(),
                                       std::vector<uint32_t>(input_shape));
  ORT_RETURN_IF_NOT(AddTensorWrapper(std::move(input_tensorwrapper)),
                    "QNN EP: Failed to add input tensor for inserted Reshape.");

  Qnn_TensorType_t tensor_type = is_for_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        tensor_data_type,
                                        quantize_param.Copy(),
                                        std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(AddTensorWrapper(std::move(output_tensorwrapper)),
                    "QNN EP: Failed to add output tensor for inserted Reshape.");

  ORT_RETURN_IF_NOT(CreateQnnNode(output_name,
                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_RESHAPE,
                                  {input_name},
                                  {output_name},
                                  {},
                                  do_op_validation),
                    "QNN EP: Failed to create manually inserted Qnn Reshape node.");

  return Status::OK();
}

Status QnnModelWrapper::AddTransposeNode(NodeIndex node_index,
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
  ORT_RETURN_IF(quantize_param.IsPerChannel(),
                "Do not support inserted Transpose nodes with per-channel quantization");
  // No need to add this for output nodes as it is added as output tensor for previous node
  if (is_for_input) {
    Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    QnnTensorWrapper input_tensorwrapper(input_name,
                                         tensor_type,
                                         tensor_data_type,
                                         quantize_param.Copy(),
                                         std::vector<uint32_t>(input_shape));
    ORT_RETURN_IF_NOT(AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  uint32_t perm_size = static_cast<uint32_t>(transpose_perm.size());
  std::vector<uint32_t> perm_dim{perm_size};
  std::vector<uint32_t> transpose_perm_copy = transpose_perm;
  const std::string& node_name = output_name;
  QnnParamWrapper transpose_param(node_index, node_name, QNN_OP_TRANSPOSE_PARAM_PERM, std::move(perm_dim), std::move(transpose_perm_copy));
  std::string param_tensor_name(transpose_param.GetParamTensorName());
  ORT_RETURN_IF_NOT(AddParamWrapper(std::move(transpose_param)), "Failed to add tensor.");
  Qnn_TensorType_t tensor_type = (false == is_for_output) ? QNN_TENSOR_TYPE_NATIVE : QNN_TENSOR_TYPE_APP_READ;
  std::vector<uint32_t> output_shape_copy = output_shape;
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        tensor_data_type,
                                        quantize_param.Copy(),
                                        std::move(output_shape_copy));
  ORT_RETURN_IF_NOT(AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  const static std::string qnn_node_type = "Transpose";

  CreateQnnNode(output_name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                qnn_node_type,
                {input_name},
                {output_name},
                {param_tensor_name},
                do_op_validation);

  return Status::OK();
}

void QnnModelWrapper::GetGraphInputOutputTensorWrapper(const std::vector<std::string>& tensor_name_list,
                                                       std::vector<QnnTensorWrapper>& wrappers_list) {
  for (const auto& tensor_name : tensor_name_list) {
    auto it = model_tensors_map_.find(tensor_name);
    if (it == model_tensors_map_.end()) {
//      LOGS(logger_, ERROR) << "Model input or output name not exist: " << tensor_name
//                           << ". Could cause execution error.";
      break;
    }
    // It's safe to move QnnTensorWrapper out of model_tensors_map_
    // since this call happens when QnnModelWrapper end of live
    wrappers_list.push_back(std::move(it->second));
    model_tensors_map_.erase(tensor_name);
  }

  return;
}

Status QnnModelWrapper::UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                                              std::vector<uint8_t>& unpacked_tensor) const {
//  if (initializer.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
//    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(initializer, graph_viewer_.ModelPath(),
//                                                                  unpacked_tensor));
//  } else {
//    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(initializer, unpacked_tensor));
//  }

  int32_t onnx_data_type = initializer.data_type();

  // If this is an int4, we need to unpack it because QNN treats int4 as a full int8.
//  if (onnx_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT4) {
//    TensorShape shape = onnxruntime::utils::GetTensorShapeFromTensorProto(initializer);
//    const size_t num_elems = shape.Size();
//    std::vector<uint8_t> packed_int4_bytes = std::move(unpacked_tensor);
//    unpacked_tensor = std::vector<uint8_t>(num_elems);
//
//    auto dst = gsl::make_span(reinterpret_cast<int8_t*>(unpacked_tensor.data()), unpacked_tensor.size());
//    auto src = gsl::make_span(reinterpret_cast<const Int4x2*>(packed_int4_bytes.data()), packed_int4_bytes.size());
//    ORT_RETURN_IF_NOT(Int4x2::Unpack(dst, src), "Failed to unpack Tensor<Int4x2> for QNN");
//  } else if (onnx_data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4) {
//    TensorShape shape = onnxruntime::utils::GetTensorShapeFromTensorProto(initializer);
//    const size_t num_elems = shape.Size();
//    std::vector<uint8_t> packed_int4_bytes = std::move(unpacked_tensor);
//    unpacked_tensor = std::vector<uint8_t>(num_elems);
//
//    auto dst = gsl::make_span(reinterpret_cast<uint8_t*>(unpacked_tensor.data()), unpacked_tensor.size());
//    auto src = gsl::make_span(reinterpret_cast<const UInt4x2*>(packed_int4_bytes.data()), packed_int4_bytes.size());
//    ORT_RETURN_IF_NOT(UInt4x2::Unpack(dst, src), "Failed to unpack Tensor<UInt4x2> for QNN");
//  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
