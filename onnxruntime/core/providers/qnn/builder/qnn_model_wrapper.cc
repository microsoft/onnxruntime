// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "qnn_model_wrapper.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace qnn {

bool QnnModelWrapper::Initialize(const Qnn_ContextHandle_t& context,
                                 const char* graph_name,
                                 bool debug,
                                 const QnnGraph_Config_t** graph_configs) {
  if (!graph_name_.empty()) {
    // only one graph is allowed per QnnModel
    LOGS(logger_, ERROR) << "Graph " << graph_name << " already initialized.";
    return false;
  }
  if (context == nullptr) {
    LOGS(logger_, ERROR) << "Nullptr passed as context.";
    return false;
  }
  if (graph_name == nullptr) {
    LOGS(logger_, ERROR) << "Nullptr passed as graphName.";
    return false;
  }

  auto rt = qnn_interface_.graphCreate(context, graph_name, graph_configs, &graph_);
  if (rt != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
    rt = qnn_interface_.graphRetrieve(context, graph_name, &graph_);
    if (rt != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
      LOGS(logger_, ERROR) << "Not able to create graph in given context.";
      return false;
    }
  }
  graph_name_ = graph_name;
  debug_ = debug;

  return true;
}

static std::vector<QnnTensorWrapper>::const_iterator QnnContainsTensorHelper(const std::string& tensor_name,
                                                                             const std::vector<QnnTensorWrapper>& tensor_wrappers) {
  return std::find_if(tensor_wrappers.cbegin(), tensor_wrappers.cend(),
                      [&tensor_name](const QnnTensorWrapper& tensor_wrapper) -> bool {
                        return tensor_wrapper.GetName() == tensor_name;
                      });
}

bool QnnModelWrapper::QnnContainsTensor(const std::string& name) const {
  return model_tensors_map_.find(name) != model_tensors_map_.end() ||
         QnnContainsTensorHelper(name, model_input_tensor_wrappers_) != model_input_tensor_wrappers_.cend() ||
         QnnContainsTensorHelper(name, model_output_tensor_wrappers_) != model_output_tensor_wrappers_.cend();
}

bool QnnModelWrapper::AddQnnTensor(const std::string& node_name,
                                   const std::string& tensor_name,
                                   const Qnn_Tensor_t& qnn_tensor,
                                   bool is_param) {
  // Verify tensor being added is not a duplicate
  if (!is_param && QnnContainsTensor(tensor_name)) {
    LOGS(logger_, VERBOSE) << "Tensor already exists: " << tensor_name
                           << ", node name: " << node_name;
    return true;
  }

  size_t data_size = utils::GetElementSizeByType(qnn_tensor.dataType);

  if (0 == data_size) {
    LOGS(logger_, ERROR) << "Invalid QNN data type provided, "
                         << qnn_tensor.dataType << ", for tensor " << tensor_name
                         << " on node " << node_name;
    return false;
  }
  {
    using namespace onnxruntime::qnn::utils;
    LOGS(logger_, VERBOSE) << "name=" << tensor_name << qnn_tensor;
  }
  // sanity check tensor data if AddTensor used for static tensor
  if (qnn_tensor.type == QNN_TENSOR_TYPE_STATIC) {
    if (qnn_tensor.memType != QNN_TENSORMEMTYPE_RAW) {
      LOGS(logger_, ERROR) << "Expected raw memType in provided static tensor "
                           << tensor_name << "for node " << node_name;
      return false;
    }
    // verify size expressed by the dims matches the raw tensor size
    uint32_t qnn_tensor_size = std::accumulate(qnn_tensor.currentDimensions,
                                               qnn_tensor.currentDimensions + qnn_tensor.rank,
                                               static_cast<uint32_t>(data_size),
                                               std::multiplies<uint32_t>());
    if (qnn_tensor_size != qnn_tensor.clientBuf.dataSize) {
      LOGS(logger_, ERROR) << "Adding STATIC tensor, length mismatch between clientBuf "
                           << "size and tensor Dims(dim) * rank * sizeof(datatype) for, node_name: " << node_name
                           << " tensor_name: " << tensor_name
                           << ". Got tensorSize: " << qnn_tensor_size
                           << ", tensor.clientBuf.dataSize: " << qnn_tensor.clientBuf.dataSize;
      return false;
    }
  }

  if (debug_ && qnn_tensor.type == QNN_TENSOR_TYPE_NATIVE) {
    // for debug, make all tensors accessible by client
    (const_cast<Qnn_Tensor_t&>(qnn_tensor)).type = QNN_TENSOR_TYPE_APP_READ;
  }

  auto tensor_create_result = qnn_interface_.tensorCreateGraphTensor(graph_, qnn_tensor);
  if (tensor_create_result != QNN_TENSOR_NO_ERROR) {
    LOGS(logger_, ERROR) << "Failed to create tensor for node: " << node_name
                         << " tensor_name: " << tensor_name
                         << " error code: " << tensor_create_result;
    return false;
  }

  return true;
}

bool QnnModelWrapper::AddTensor(const std::string& node_name,
                                QnnTensorWrapper&& tensor_wrapper) {
  std::string tensor_name = tensor_wrapper.GetName();
  if (QnnContainsTensor(tensor_name) == true) {
    return true;
  }
  if (AddQnnTensor(node_name, tensor_name, tensor_wrapper.GetQnnTensor()) == false) {
    return false;
  }
  const Qnn_TensorType_t& qnn_tensor_type = tensor_wrapper.GetQnnTensor().type;
  // save network input/outputs tensors to use for setting the Qnn graph's input and output
  // tensors for populating GraphInfo for caller
  if (qnn_tensor_type == QNN_TENSOR_TYPE_APP_WRITE) {
    model_input_tensor_wrappers_.push_back(std::move(tensor_wrapper));
  } else if (qnn_tensor_type == QNN_TENSOR_TYPE_APP_READ) {
    model_output_tensor_wrappers_.push_back(std::move(tensor_wrapper));
  } else {
    // save created tensors for later lookup to populate graph node construction
    model_tensors_map_.emplace(tensor_name, std::move(tensor_wrapper));
  }

  return true;
}

bool QnnModelWrapper::GetQnnTensor(const std::string& tensor_name, Qnn_Tensor_t& tensor) {
  auto map_iter = model_tensors_map_.find(tensor_name);
  if (map_iter != model_tensors_map_.end()) {
    tensor = map_iter->second.GetQnnTensor();
    return true;
  } else {
    auto input_iter = QnnContainsTensorHelper(tensor_name, model_input_tensor_wrappers_);
    if (input_iter != model_input_tensor_wrappers_.cend()) {
      tensor = input_iter->GetQnnTensor();
      return true;
    } else {
      auto output_iter = QnnContainsTensorHelper(tensor_name, model_output_tensor_wrappers_);
      if (output_iter != model_output_tensor_wrappers_.cend()) {
        tensor = output_iter->GetQnnTensor();
        return true;
      }
    }
  }
  return false;
}

bool QnnModelWrapper::AddParams(const std::string& node_name,
                                const std::vector<QnnParamWrapper>& param_wrappers) {
  bool rt = true;
  for (const QnnParamWrapper& param_wrapper : param_wrappers) {
    const Qnn_Param_t& param = param_wrapper.GetQnnParam();
    switch (param.paramType) {
      case QNN_PARAMTYPE_TENSOR: {
        LOGS(logger_, VERBOSE) << "Add parameter tensor: " << param.name;
        rt = AddQnnTensor(node_name, param.name, param.tensorParam, true);
        if (!rt) {
          LOGS(logger_, ERROR) << "AddTensor failed for tensor param: "
                               << param.name << " on node: " << node_name;
          return rt;
        }
        break;
      }
      case QNN_PARAMTYPE_SCALAR: {
        LOGS(logger_, VERBOSE) << "Add parameter scalar: " << param.name;
        break;
      }
      default: {
        LOGS(logger_, ERROR) << "Unknown param type passed for param: "
                             << param.name << " on node: " << node_name;
        return false;
      }
    }
  }
  return true;
}

bool QnnModelWrapper::AddNode(const std::string& qnn_node_name,
                              const std::string& package_name,
                              const std::string& qnn_node_type,
                              std::vector<QnnParamWrapper>&& params,
                              const std::vector<std::string>& input_names,
                              std::vector<QnnTensorWrapper>&& output_wrappers,
                              bool do_op_validation) {
  bool rt = AddParams(qnn_node_name, params);

  std::vector<Qnn_Param_t> qnn_params;
  // params_.insert(std::end(params_), std::make_move_iterator(params.begin()), std::make_move_iterator(params.end()));
  std::transform(params.begin(), params.end(), std::back_inserter(qnn_params),
                 [](QnnParamWrapper& param) -> const Qnn_Param_t& { return param.GetQnnParam(); });
  // populate input tensors for node
  auto num_of_inputs = static_cast<uint32_t>(input_names.size());
  std::vector<Qnn_Tensor_t> inputs(num_of_inputs, QNN_TENSOR_INIT);
  for (size_t j = 0; j < num_of_inputs; j++) {
    rt = GetQnnTensor(input_names[j], inputs[j]);
    if (!rt) {
      LOGS(logger_, ERROR) << "GetQnnTensor failed for tensor: "
                           << input_names[j] << " on node: " << qnn_node_name;
      return rt;
    }
  }

  // populate output tensors of node
  auto num_of_outputs = static_cast<uint32_t>(output_wrappers.size());
  std::vector<Qnn_Tensor_t> outputs(num_of_outputs, QNN_TENSOR_INIT);
  for (size_t k = 0; k < num_of_outputs; k++) {
    // create node output tensors first
    std::string output_name = output_wrappers[k].GetName();
    LOGS(logger_, VERBOSE) << "Add output: " << output_name;
    rt = AddTensor(qnn_node_name, std::move(output_wrappers[k]));
    if (!rt) {
      LOGS(logger_, ERROR) << "AddTensor failed for tensor: "
                           << output_name << " on node: " << qnn_node_name;
      return rt;
    }
    rt = GetQnnTensor(output_name, outputs[k]);
    if (!rt) {
      LOGS(logger_, ERROR) << "getQnnTensor failed for tensor: "
                           << output_name << " on node: " << qnn_node_name;
      return rt;
    }
  }

  QnnOpConfigWrapper op_config_wrapper(qnn_node_name,
                                       package_name,
                                       qnn_node_type,
                                       std::move(qnn_params),
                                       std::move(inputs),
                                       std::move(outputs));
  const Qnn_OpConfig_t& op_config = op_config_wrapper.GetQnnOpConfig();
  using namespace onnxruntime::qnn::utils;
  LOGS(logger_, VERBOSE) << op_config;

  if (do_op_validation) {
    auto validation_status = qnn_interface_.backendValidateOpConfig(op_config);
    if (QNN_SUCCESS != validation_status) {
      LOGS(logger_, WARNING) << "Validating node failed for: " << qnn_node_name;
      return false;
    } else {
      return true;
    }
  }
  if (qnn_interface_.graphAddNode(graph_, op_config) != QNN_GRAPH_NO_ERROR) {
    LOGS(logger_, ERROR) << "Adding node failed for: " << qnn_node_name;
    return false;
  }

  return true;
}

bool QnnModelWrapper::GetOnnxShape(const NodeArg& node_arg, std::vector<uint32_t>& shape) {
  const auto* shape_proto = node_arg.Shape();
  if (shape_proto == nullptr) {
    return false;
  }

  // For Scalar data, we need to set shape to 1 for QNN
  if (shape_proto->dim_size() < 1) {
    shape.push_back(1);
    return true;
  }

  // We already checked the shape has no dynamic dimension
  for (const auto& dim : shape_proto->dim()) {
    shape.push_back(SafeInt<uint32_t>(dim.dim_value()));
  }

  return true;
}

bool QnnModelWrapper::ProcessOffset(const std::string& offset_name,
                                    int32_t& offset_value) const {
  const auto& graph_initializers = GetInitializerTensors();
  auto offset_it = graph_initializers.find(offset_name);
  if (offset_it == graph_initializers.end()) {
    LOGS(logger_, ERROR) << "Not able to find initializer: " << offset_name;
    return false;
  }
  const auto offset_tensor = offset_it->second;
  const int32_t onnx_data_type = offset_tensor->data_type();

  std::vector<uint8_t> unpacked_tensor;
  ORT_THROW_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*offset_tensor, unpacked_tensor));
  switch (onnx_data_type) {
    // QNN use -offest for some reason
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      auto int8_span = gsl::make_span(unpacked_tensor).as_span<const int8_t>();
      offset_value = -(int8_span.data()[0]);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      auto uint8_span = gsl::make_span(unpacked_tensor).as_span<const uint8_t>();
      offset_value = 0 - (uint8_span.data()[0]);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      auto int32_span = gsl::make_span(unpacked_tensor).as_span<const int32_t>();
      offset_value = -(int32_span.data()[0]);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
      auto uint32_span = gsl::make_span(unpacked_tensor).as_span<const uint32_t>();
      offset_value = 0 - (uint32_span.data()[0]);
      break;
    }
    default: {
      LOGS(logger_, ERROR) << "Data type not supported!";
      return false;
    }
  }
  return true;
}

bool QnnModelWrapper::ProcessScale(const std::string& scale_name,
                                   float& scale_value) const {
  const auto& graph_initializers = GetInitializerTensors();
  auto offset_it = graph_initializers.find(scale_name);
  if (offset_it == graph_initializers.end()) {
    LOGS(logger_, ERROR) << "Not able to find initializer: " << scale_name;
    return false;
  }
  const auto scale_tensor = offset_it->second;
  std::vector<uint8_t> unpacked_tensor;

  ORT_THROW_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*scale_tensor, unpacked_tensor));
  const float* scale_data = reinterpret_cast<float*>(unpacked_tensor.data());
  scale_value = scale_data[0];
  return true;
}

bool QnnModelWrapper::ProcessQuantizationParameter(const std::optional<NodeUnitIODef::QuantParam>& quant_param,
                                                   float& scale_value,
                                                   int32_t& offset_value) const {
  if (quant_param.has_value()) {
    // Parse scale & zero_point
    const auto& scale_name = quant_param->scale.Name();
    bool rt = ProcessScale(scale_name, scale_value);
    if (!rt) {
      return rt;
    }

    if (quant_param->zero_point) {
      const auto& zero_point_name = quant_param->zero_point->Name();
      return ProcessOffset(zero_point_name, offset_value);
    }
  }
  return true;
}

Status QnnModelWrapper::AddTransposeNode(NodeIndex node_index,
                                         const std::string& input_name,
                                         const std::string& output_name,
                                         const std::vector<uint32_t>& input_shape,
                                         const std::vector<uint32_t>& transpose_perm,
                                         const std::vector<uint32_t>& output_shape,
                                         const Qnn_DataType_t& tensor_data_type,
                                         const Qnn_QuantizeParams_t& quantize_param,
                                         const bool is_for_input,
                                         const bool is_for_output) {
  // No need to add this for output nodes as it is added as output tensor for previous node
  if (is_for_input) {
    Qnn_TensorDataFormat_t data_format = 0;
    Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    QnnTensorWrapper input_tensorwrapper(input_name,
                                         tensor_type,
                                         data_format,
                                         tensor_data_type,
                                         quantize_param,
                                         std::vector<uint32_t>(input_shape));
    AddTensor(input_name, std::move(input_tensorwrapper));
  }
  std::vector<std::string> input_names{input_name};

  uint32_t perm_size = static_cast<uint32_t>(transpose_perm.size());
  std::vector<uint32_t> perm_dim{perm_size};
  std::vector<uint32_t> transpose_perm_copy = transpose_perm;
  const std::string& node_name = output_name;
  QnnParamWrapper transpose_param(node_index, node_name, qnn_def::perm, std::move(perm_dim), std::move(transpose_perm_copy));
  Qnn_TensorType_t tensor_type = (false == is_for_output) ? QNN_TENSOR_TYPE_NATIVE : QNN_TENSOR_TYPE_APP_READ;
  Qnn_TensorDataFormat_t data_format = 0;
  std::vector<uint32_t> output_shape_copy = output_shape;
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        data_format,
                                        tensor_data_type,
                                        quantize_param,
                                        std::move(output_shape_copy));
  const static std::string qnn_node_type = "Transpose";
  std::vector<QnnParamWrapper> params;
  params.push_back(std::move(transpose_param));
  std::vector<QnnTensorWrapper> outputs;
  outputs.push_back(std::move(output_tensorwrapper));

  AddNode(output_name,            // Node Name
          qnn_def::package_name,  // Package Name
          qnn_node_type,          // Qnn Node Type
          std::move(params),      // Node Params
          input_names,            // Input Tensor Names
          std::move(outputs)      // Output Tensors
  );

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
