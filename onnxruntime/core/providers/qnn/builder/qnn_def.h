// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "QnnInterface.h"
#include "qnn_utils.h"
#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include <climits>
#include <type_traits>
#include "core/graph/basic_types.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace qnn {

enum ProfilingLevel { OFF,
                      BASIC,
                      DETAILED,
                      INVALID };

const size_t kbitsPerByte = CHAR_BIT;

struct OnnxTensorInfo {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxTensorInfo);
  OnnxTensorInfo(size_t index, int32_t data_type, const std::vector<int64_t>&& shape) : index_(index), data_type_(data_type), shape_(std::move(shape)) {}
  size_t index_;
  const int32_t data_type_;  // Uses TensorProto::DataType
  const std::vector<int64_t> shape_;
};

size_t memscpy(void* dst, size_t dstSize, const void* src, size_t copySize);

class QnnTensorWrapper {
 public:
  QnnTensorWrapper(const std::string& name,
                   Qnn_TensorType_t tensor_type,
                   Qnn_TensorDataFormat_t data_format,
                   Qnn_DataType_t data_type,
                   const Qnn_QuantizeParams_t& quantize_params,
                   std::vector<uint32_t>&& shape,
                   std::vector<uint8_t>&& client_buf = {},
                   Qnn_TensorMemType_t mem_type = QNN_TENSORMEMTYPE_RAW) : tensor_name_(name),
                                                                           max_dimensions_(std::move(shape)),
                                                                           current_dimensions_(max_dimensions_.size(), 0),
                                                                           client_buf_(std::move(client_buf)) {
    if (tensor_type == QNN_TENSOR_TYPE_STATIC) {
      current_dimensions_ = max_dimensions_;
    }
    qnn_tensor_.id = utils::GetTensorIdFromName(tensor_name_);
    qnn_tensor_.type = tensor_type;
    qnn_tensor_.dataFormat = data_format;
    qnn_tensor_.dataType = data_type;
    qnn_tensor_.rank = static_cast<uint32_t>(max_dimensions_.size());
    qnn_tensor_.maxDimensions = max_dimensions_.data();
    qnn_tensor_.currentDimensions = current_dimensions_.data();
    qnn_tensor_.memType = mem_type;
    if (mem_type == QNN_TENSORMEMTYPE_RAW) {
      auto size = client_buf_.size() * sizeof(uint8_t);
      qnn_tensor_.clientBuf.data = client_buf_.data();
      qnn_tensor_.clientBuf.dataSize = static_cast<uint32_t>(size);
    } else if (mem_type == QNN_TENSORMEMTYPE_MEMHANDLE) {
      qnn_tensor_.memHandle = nullptr;
    } else {
      ORT_THROW("Unexpected mem_type for tensor");
    }
    CopyQuantizationEncoding(qnn_tensor_.quantizeParams, quantize_params);
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnTensorWrapper);

  QnnTensorWrapper(QnnTensorWrapper&& other) noexcept {
    std::swap(qnn_tensor_, other.qnn_tensor_);
    std::swap(tensor_name_, other.tensor_name_);
    std::swap(client_buf_, other.client_buf_);
    std::swap(max_dimensions_, other.max_dimensions_);
    std::swap(current_dimensions_, other.current_dimensions_);
    qnn_tensor_.maxDimensions = max_dimensions_.data();
    qnn_tensor_.currentDimensions = current_dimensions_.data();
    qnn_tensor_.clientBuf.data = client_buf_.data();
  }

  ~QnnTensorWrapper() = default;

  const Qnn_Tensor_t& GetQnnTensor() const {
    return qnn_tensor_;
  }
  const std::string& GetName() const { return tensor_name_; }

 private:
  void CopyQuantizationEncoding(Qnn_QuantizeParams_t& dst, const Qnn_QuantizeParams_t& src) {
    Qnn_QuantizationEncoding_t encoding = src.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
        encoding == QNN_QUANTIZATION_ENCODING_UNDEFINED) {
      dst = src;
    } else if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      ORT_THROW("Axis scale offset quantization parameter is not supported.");
    } else {
      ORT_THROW("quantizationEncoding incorrect value.");
    }
  }

  std::string tensor_name_;
  std::vector<uint32_t> max_dimensions_;
  std::vector<uint32_t> current_dimensions_;
  std::vector<uint8_t> client_buf_;
  Qnn_Tensor_t qnn_tensor_ = QNN_TENSOR_INIT;
};

class QnnParamWrapper {
 public:
  QnnParamWrapper(const std::string& name,
                  Qnn_Scalar_t scalarParam) : name_(name), shape_({}), param_data_({}) {
    qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
    qnn_param_.name = name_.c_str();
    qnn_param_.scalarParam = scalarParam;
  }

  QnnParamWrapper(NodeIndex node_index,
                  const std::string& node_name,
                  const std::string& name,
                  std::vector<uint32_t>&& shape,
                  std::vector<uint32_t>&& param_data,
                  bool is_signed = false) : name_(name), shape_(std::move(shape)), param_data_(std::move(param_data)) {
    qnn_param_.paramType = QNN_PARAMTYPE_TENSOR;
    qnn_param_.name = name_.c_str();
    std::stringstream ss;
    ss << node_name << "_" << node_index << "_" << name;
    qnn_param_.tensorParam = QNN_TENSOR_INIT;
    qnn_param_.tensorParam.id = utils::GetTensorIdFromName(ss.str());
    qnn_param_.tensorParam.type = QNN_TENSOR_TYPE_STATIC;
    qnn_param_.tensorParam.dataType = is_signed ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_UINT_32;
    qnn_param_.tensorParam.rank = static_cast<uint32_t>(shape_.size());
    qnn_param_.tensorParam.memType = QNN_TENSORMEMTYPE_RAW;
    qnn_param_.tensorParam.maxDimensions = shape_.data();
    qnn_param_.tensorParam.currentDimensions = shape_.data();
    qnn_param_.tensorParam.clientBuf.data = param_data_.data();
    // TODO Need to convert the data from unsigned to signed.
    qnn_param_.tensorParam.clientBuf.dataSize = static_cast<uint32_t>(param_data_.size() * sizeof(uint32_t));
    qnn_param_.tensorParam.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnParamWrapper);
  QnnParamWrapper(QnnParamWrapper&& other) noexcept {
    std::swap(name_, other.name_);
    std::swap(shape_, other.shape_);
    std::swap(param_data_, other.param_data_);
    std::swap(qnn_param_, other.qnn_param_);
    qnn_param_.name = name_.c_str();
    if (qnn_param_.paramType == QNN_PARAMTYPE_TENSOR) {
      qnn_param_.tensorParam.maxDimensions = shape_.data();
      qnn_param_.tensorParam.currentDimensions = shape_.data();
      qnn_param_.tensorParam.clientBuf.data = param_data_.data();
    }
  }

  ~QnnParamWrapper() = default;

  const char* GetName() const {
    return qnn_param_.name;
  }

  const Qnn_Param_t& GetQnnParam() const {
    return qnn_param_;
  }

 private:
  std::string name_;
  std::vector<uint32_t> shape_;
  std::vector<uint32_t> param_data_;
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;
};

class QnnOpConfigWrapper {
 public:
  QnnOpConfigWrapper(const std::string& name,
                     const std::string& package_name,
                     const std::string& type_name,
                     std::vector<Qnn_Param_t>&& params,
                     std::vector<Qnn_Tensor_t>&& inputs,
                     std::vector<Qnn_Tensor_t>&& outputs) : name_(name),
                                                            package_name_(package_name),
                                                            type_name_(type_name),
                                                            params_(std::move(params)),
                                                            inputs_(std::move(inputs)),
                                                            outputs_(std::move(outputs)) {
    op_config_.name = name_.c_str();
    op_config_.packageName = package_name_.c_str();
    op_config_.typeName = type_name_.c_str();
    op_config_.numOfParams = static_cast<uint32_t>(params_.size());
    op_config_.numOfInputs = static_cast<uint32_t>(inputs_.size());
    op_config_.numOfOutputs = static_cast<uint32_t>(outputs_.size());
    op_config_.params = params_.data();
    op_config_.inputTensors = inputs_.data();
    op_config_.outputTensors = outputs_.data();
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnOpConfigWrapper);

  QnnOpConfigWrapper(QnnOpConfigWrapper&& other) noexcept {
    std::swap(this->op_config_, other.op_config_);
    std::swap(name_, other.name_);
    std::swap(package_name_, other.package_name_);
    std::swap(type_name_, other.type_name_);
    std::swap(params_, other.params_);
    std::swap(inputs_, other.inputs_);
    std::swap(outputs_, other.outputs_);
    op_config_.name = name_.c_str();
    op_config_.packageName = package_name_.c_str();
    op_config_.typeName = type_name_.c_str();
    op_config_.params = params_.data();
    op_config_.inputTensors = inputs_.data();
    op_config_.outputTensors = outputs_.data();
  }

  ~QnnOpConfigWrapper() = default;

  const Qnn_OpConfig_t& GetQnnOpConfig() { return op_config_; }

 private:
  std::string name_;
  std::string package_name_;
  std::string type_name_;
  std::vector<Qnn_Param_t> params_;
  std::vector<Qnn_Tensor_t> inputs_;
  std::vector<Qnn_Tensor_t> outputs_;
  Qnn_OpConfig_t op_config_ = QNN_OPCONFIG_INIT;
};

class GraphInfo {
 public:
  GraphInfo(Qnn_GraphHandle_t graph, const std::string& name,
            std::vector<QnnTensorWrapper>&& input_tensors,
            std::vector<QnnTensorWrapper>&& output_tensors,
            std::unordered_map<std::string, QnnTensorWrapper>&& model_tensor_map) : graph_name_(name),
                                                                                    graph_(graph),
                                                                                    input_tensors_(std::move(input_tensors)),
                                                                                    output_tensors_(std::move(output_tensors)),
                                                                                    model_tensors_map_(std::move(model_tensor_map)) {
  }
  size_t NumInputTensors() const { return input_tensors_.size(); }
  size_t NumOutputTensors() const { return output_tensors_.size(); }
  const std::string& Name() const { return graph_name_; }
  const std::vector<QnnTensorWrapper>& InputTensors() const { return input_tensors_; }
  const std::vector<QnnTensorWrapper>& OutputTensors() const { return output_tensors_; }
  const Qnn_GraphHandle_t& Graph() const { return graph_; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphInfo);

 private:
  std::string graph_name_;
  Qnn_GraphHandle_t graph_;
  std::vector<QnnTensorWrapper> input_tensors_;
  std::vector<QnnTensorWrapper> output_tensors_;
  std::unordered_map<std::string, QnnTensorWrapper> model_tensors_map_;
};

typedef GraphInfo* GraphInfoPtr_t;

typedef struct GraphConfigInfo {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphConfigInfo);
  const char* graphName;
  const QnnGraph_Config_t** graphConfigs;
} GraphConfigInfo_t;

void QnnLogStdoutCallback(const char* format,
                          QnnLog_Level_t level,
                          uint64_t timestamp,
                          va_list argument_parameter);
static std::mutex qnn_log_mutex_;

namespace qnn_def {
const std::string package_name = "qti.aisw";
const std::string dilation = "dilation";
const std::string pad_amount = "pad_amount";
const std::string stride = "stride";
const std::string group = "group";
const std::string filter_size = "filter_size";
const std::string count_pad_for_edges = "count_pad_for_edges";
const std::string perm = "perm";
const std::string axis = "axis";
const std::string axes = "axes";
const std::string keep_dims = "keep_dims";
const std::string transpose_in0 = "transpose_in0";
const std::string transpose_in1 = "transpose_in1";
const std::string min_value = "min_value";
const std::string max_value = "max_value";
const std::string ranges = "ranges";
const std::string output_padding = "output_padding";
const std::string split_index = "split_index";
const std::string align_corners = "align_corners";
const std::string half_pixel_centers = "half_pixel_centers";
const std::string exclude_outside = "exclude_outside";
const std::string transformation_mode = "transformation_mode";
const std::string interpolation_mode = "interpolation_mode";
const std::string nearest_mode = "nearest_mode";
const std::string rounding_mode = "rounding_mode";
}  // namespace qnn_def

}  // namespace qnn
}  // namespace onnxruntime
