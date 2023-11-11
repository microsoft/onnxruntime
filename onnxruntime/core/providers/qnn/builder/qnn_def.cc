// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include <memory>
#include <ostream>
#include <cstring>

namespace onnxruntime {
namespace qnn {

size_t memscpy(void* dst, size_t dst_size, const void* src, size_t copy_size) {
  if (!dst || !src || !dst_size || !copy_size) return 0;

  size_t min_size = dst_size < copy_size ? dst_size : copy_size;

  memcpy(dst, src, min_size);

  return min_size;
}

void SetQnnTensorType(Qnn_Tensor_t& qnn_tensor, Qnn_TensorType_t tensor_type) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.type = tensor_type;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorName(Qnn_Tensor_t& qnn_tensor, const char* name) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.name = name;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorDataFormat(Qnn_Tensor_t& qnn_tensor, Qnn_TensorDataFormat_t data_format) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.dataFormat = data_format;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorDataType(Qnn_Tensor_t& qnn_tensor, Qnn_DataType_t data_type) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.dataType = data_type;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorDim(Qnn_Tensor_t& qnn_tensor, const std::vector<uint32_t>& dimensions) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.rank = static_cast<uint32_t>(dimensions.size());
    qnn_tensor.v1.dimensions = const_cast<uint32_t*>(dimensions.data());
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorMemType(Qnn_Tensor_t& qnn_tensor, Qnn_TensorMemType_t mem_type) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.memType = mem_type;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorClientBuf(Qnn_Tensor_t& qnn_tensor, const std::vector<uint8_t>& client_buf) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    auto size = client_buf.size() * sizeof(uint8_t);
    qnn_tensor.v1.clientBuf.data = const_cast<void*>(static_cast<const void*>(client_buf.data()));
    qnn_tensor.v1.clientBuf.dataSize = static_cast<uint32_t>(size);
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorClientBuf(Qnn_Tensor_t& qnn_tensor, const std::vector<uint32_t>& client_buf) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    auto size = client_buf.size() * sizeof(uint32_t);
    qnn_tensor.v1.clientBuf.data = const_cast<void*>(static_cast<const void*>(client_buf.data()));
    qnn_tensor.v1.clientBuf.dataSize = static_cast<uint32_t>(size);
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorClientBufSize(Qnn_Tensor_t& qnn_tensor, uint32_t client_buf_size) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.clientBuf.dataSize = client_buf_size;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorClientBufData(Qnn_Tensor_t& qnn_tensor, void* client_buf_data) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    qnn_tensor.v1.clientBuf.data = client_buf_data;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

void SetQnnTensorQParams(Qnn_Tensor_t& qnn_tensor, const Qnn_QuantizeParams_t& quantize_params) {
  Qnn_QuantizationEncoding_t encoding = quantize_params.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
      encoding == QNN_QUANTIZATION_ENCODING_UNDEFINED) {
    if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
      qnn_tensor.v1.quantizeParams = quantize_params;
    } else {
      ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
    }
  } else if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    ORT_THROW("Axis scale offset quantization parameter is not supported.");
  } else {
    ORT_THROW("quantizationEncoding incorrect value.");
  }
}

uint32_t GetQnnTensorID(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.id;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

Qnn_TensorType_t GetQnnTensorType(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.type;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

const char* GetQnnTensorName(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.name;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

Qnn_TensorDataFormat_t GetQnnTensorDataFormat(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.dataFormat;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

Qnn_DataType_t GetQnnTensorDataType(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.dataType;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

Qnn_TensorMemType_t GetQnnTensorMemType(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.memType;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

uint32_t GetQnnTensorRank(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.rank;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

uint32_t* GetQnnTensorDims(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.dimensions;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

const Qnn_ClientBuffer_t& GetQnnTensorClientBuf(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.clientBuf;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

const Qnn_QuantizeParams_t& GetQnnTensorQParams(const Qnn_Tensor_t& qnn_tensor) {
  if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
    return qnn_tensor.v1.quantizeParams;
  } else {
    ORT_THROW("QNN tensor version not supported, QNN tensor version: ", qnn_tensor.version);
  }
}

Status CompareQnnQuantParams(const Qnn_QuantizeParams_t& qparam0, const Qnn_QuantizeParams_t& qparam1,
                             float& scale_diff, int32_t& offset_diff) {
  scale_diff = 0.0f;
  offset_diff = 0;

  ORT_RETURN_IF_NOT((qparam0.encodingDefinition == qparam1.encodingDefinition &&
                     qparam0.quantizationEncoding == qparam1.quantizationEncoding),
                    "Expected quantization parameters to be the same type.");

  if (qparam0.encodingDefinition == QNN_DEFINITION_DEFINED) {
    switch (qparam0.quantizationEncoding) {
      case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
        scale_diff = std::abs(qparam0.scaleOffsetEncoding.scale - qparam1.scaleOffsetEncoding.scale);
        offset_diff = std::abs(qparam0.scaleOffsetEncoding.offset - qparam1.scaleOffsetEncoding.offset);
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported quantization encoding: ", qparam0.quantizationEncoding);
    }
  }

  return Status::OK();
}

bool CreateTensorInQnnGraph(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                            const Qnn_GraphHandle_t& graph,
                            const std::string& node_name,
                            const std::string& tensor_name,
                            Qnn_Tensor_t& qnn_tensor,
                            std::unordered_map<std::string, bool>& tensors_created_table,
                            std::string& error_msg) {
  if (tensors_created_table.find(tensor_name) != tensors_created_table.end()) {
    error_msg = "Tensor created already: " + tensor_name;
    return true;
  }

  auto qnn_data_type = GetQnnTensorDataType(qnn_tensor);
  size_t data_size = utils::GetElementSizeByType(qnn_data_type);

  std::stringstream ss;
  if (0 == data_size) {
    ss << "Invalid QNN data type provided, "
       << qnn_data_type << ", for tensor " << tensor_name
       << " on node " << node_name;
    error_msg = ss.str();
    return false;
  }

  // sanity check tensor data if AddTensor used for static tensor
  auto qnn_tensor_type = GetQnnTensorType(qnn_tensor);
  if (qnn_tensor_type == QNN_TENSOR_TYPE_STATIC) {
    if (GetQnnTensorMemType(qnn_tensor) != QNN_TENSORMEMTYPE_RAW) {
      ss << "Expected raw memType in provided static tensor "
         << tensor_name << "for node " << node_name;
      error_msg = ss.str();
      return false;
    }
    // verify size expressed by the dims matches the raw tensor size
    auto qnn_tensor_dims = GetQnnTensorDims(qnn_tensor);
    auto qnn_tensor_rank = GetQnnTensorRank(qnn_tensor);
    uint32_t qnn_tensor_size = std::accumulate(qnn_tensor_dims,
                                               qnn_tensor_dims + qnn_tensor_rank,
                                               static_cast<uint32_t>(data_size),
                                               std::multiplies<uint32_t>());
    auto qnn_tensor_buf_size = GetQnnTensorClientBuf(qnn_tensor).dataSize;
    if (qnn_tensor_size != qnn_tensor_buf_size) {
      ss << "Data length mismatch for static tensor. node_name: " << node_name
         << " tensor_name: " << tensor_name
         << ". size calculated from shape: " << qnn_tensor_size
         << ", tensor.clientBuf.dataSize: " << qnn_tensor_buf_size;
      error_msg = ss.str();
      return false;
    }
  }

  auto tensor_create_result = qnn_interface.tensorCreateGraphTensor(graph, &qnn_tensor);
  if (tensor_create_result != QNN_TENSOR_NO_ERROR) {
    ss << "Failed to create tensor for node: " << node_name
       << " tensor_name: " << tensor_name
       << " error code: " << tensor_create_result;
    error_msg = ss.str();
    return false;
  }

  tensors_created_table.emplace(tensor_name, true);
  return true;
}

bool QnnParamWrapper::CreateQnnGraphParam(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                          const Qnn_GraphHandle_t& graph,
                                          const std::string& node_name,
                                          std::unordered_map<std::string, bool>& tensors_created_table,
                                          std::string& error_msg) {
  std::stringstream ss;
  switch (qnn_param_.paramType) {
    case QNN_PARAMTYPE_TENSOR: {
      return CreateTensorInQnnGraph(qnn_interface, graph, node_name, tensor_name_,
                                    qnn_param_.tensorParam, tensors_created_table, error_msg);
    }
    case QNN_PARAMTYPE_SCALAR: {
      ss << "Add scalar parameter: " << name_;
      error_msg = ss.str();
      return true;
    }
    default: {
      ss << "Unknown param type passed for param: "
         << name_ << " on node: " << node_name;
      error_msg = ss.str();
      return true;
    }
  }

  return true;
}

void QnnOpConfigWrapper::SetNames(const char* op_name,
                                  const char* package_name,
                                  const char* type_name) {
  if (QNN_OPCONFIG_VERSION_1 == op_config_.version) {
    op_config_.v1.name = op_name;
    op_config_.v1.packageName = package_name;
    op_config_.v1.typeName = type_name;
  } else {
    ORT_THROW("QNN OpConfig version not supported, QNN OpConfig version: ", op_config_.version);
  }
}

void QnnOpConfigWrapper::SetNums(uint32_t num_inputs,
                                 uint32_t num_outputs,
                                 uint32_t num_params) {
  if (QNN_OPCONFIG_VERSION_1 == op_config_.version) {
    op_config_.v1.numOfInputs = num_inputs;
    op_config_.v1.numOfOutputs = num_outputs;
    op_config_.v1.numOfParams = num_params;
  } else {
    ORT_THROW("QNN OpConfig version not supported, QNN OpConfig version: ", op_config_.version);
  }
}

void QnnOpConfigWrapper::SetData(Qnn_Tensor_t* input_tensors,
                                 Qnn_Tensor_t* output_tensors,
                                 Qnn_Param_t* params) {
  if (QNN_OPCONFIG_VERSION_1 == op_config_.version) {
    op_config_.v1.inputTensors = input_tensors;
    op_config_.v1.outputTensors = output_tensors;
    op_config_.v1.params = params;
  } else {
    ORT_THROW("QNN OpConfig version not supported, QNN OpConfig version: ", op_config_.version);
  }
}

bool QnnOpConfigWrapper::QnnGraphOpValidation(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                              const Qnn_BackendHandle_t& backend_handle,
                                              std::string& error_msg) {
  auto validation_status = qnn_interface.backendValidateOpConfig(backend_handle, op_config_);
  if (QNN_SUCCESS != validation_status) {
    std::ostringstream oss;
    oss << "QNN.backendValidateOpConfig() failed for node `" << name_ << "` of type `"
        << type_name_ << "` with error code " << validation_status << std::endl;
    error_msg = oss.str();
    return false;
  }

  return true;
}

bool QnnOpConfigWrapper::CreateQnnGraphOp(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                          const Qnn_GraphHandle_t& graph,
                                          std::string& error_msg) {
  auto status = qnn_interface.graphAddNode(graph, op_config_);
  if (QNN_GRAPH_NO_ERROR != status) {
    std::ostringstream oss;
    oss << "QNN.graphAddNode() failed for node `" << name_ << "` of type `" << type_name_
        << "` with error code " << status << std::endl;
    error_msg = oss.str();
    return false;
  }

  return true;
}

bool IsNpuBackend(QnnBackendType backend_type) {
  return backend_type == QnnBackendType::HTP || backend_type == QnnBackendType::DSP;
}

}  // namespace qnn
}  // namespace onnxruntime
