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

#include "core/framework/allocator.h"
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
  OnnxTensorInfo(int32_t data_type, const std::vector<int64_t>& shape) : data_type_(data_type), shape_(shape) {}
  const int32_t data_type_;  // Uses TensorProto::DataType
  const std::vector<int64_t> shape_;
};

size_t memscpy(void* dst, size_t dstSize, const void* src, size_t copySize);

class QnnTensorWrapper {
 public:
  QnnTensorWrapper(onnxruntime::AllocatorPtr cpu_allocator,
                   const std::string& name,
                   Qnn_TensorType_t tensor_type,
                   Qnn_TensorDataFormat_t data_format,
                   Qnn_DataType_t data_type,
                   const Qnn_QuantizeParams_t& quantize_params,
                   std::vector<uint32_t>&& shape,
                   std::vector<uint8_t>&& client_buf = {},
                   Qnn_TensorMemType_t mem_type = QNN_TENSORMEMTYPE_RAW) : tensor_name_(name), cpu_allocator_(cpu_allocator) {
    qnn_tensor_.id = utils::GetTensorIdFromName(tensor_name_);
    qnn_tensor_.type = tensor_type;
    qnn_tensor_.dataFormat = data_format;
    qnn_tensor_.dataType = data_type;
    qnn_tensor_.rank = static_cast<uint32_t>(shape.size());
    {
      auto size = shape.size() * sizeof(uint32_t);
      qnn_tensor_.maxDimensions = reinterpret_cast<uint32_t*>(cpu_allocator_->Alloc(size));
      qnn_tensor_.currentDimensions = reinterpret_cast<uint32_t*>(cpu_allocator_->Alloc(size));
      memscpy(qnn_tensor_.maxDimensions, size, shape.data(), size);
      memscpy(qnn_tensor_.currentDimensions, size, shape.data(), size);
    }
    qnn_tensor_.memType = mem_type;
    if (mem_type == QNN_TENSORMEMTYPE_RAW) {
      if (client_buf.size() > 0) {
        auto size = client_buf.size() * sizeof(uint8_t);
        qnn_tensor_.clientBuf.data = reinterpret_cast<int8_t*>(cpu_allocator_->Alloc(size));
        memscpy(qnn_tensor_.clientBuf.data, size, client_buf.data(), size);
        qnn_tensor_.clientBuf.dataSize = static_cast<uint32_t>(size * sizeof(uint8_t));
      } else {
        qnn_tensor_.clientBuf.data = nullptr;
        qnn_tensor_.clientBuf.dataSize = 0;
      }
    } else if (mem_type == QNN_TENSORMEMTYPE_MEMHANDLE) {
      qnn_tensor_.memHandle = nullptr;
    } else {
      ORT_THROW("Unexpected mem_type for tensor");
    }
    CopyQuantizationEncoding(qnn_tensor_.quantizeParams, quantize_params);
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnTensorWrapper);

  QnnTensorWrapper(QnnTensorWrapper&& other) noexcept {
    std::swap(this->qnn_tensor_, other.qnn_tensor_);
    std::swap(tensor_name_, other.tensor_name_);
    std::swap(cpu_allocator_, other.cpu_allocator_);
  }

  ~QnnTensorWrapper() {
    if (nullptr != cpu_allocator_) {
      if (nullptr != qnn_tensor_.maxDimensions) {
        cpu_allocator_->Free(qnn_tensor_.maxDimensions);
        qnn_tensor_.maxDimensions = nullptr;
      }

      if (nullptr != qnn_tensor_.currentDimensions) {
        cpu_allocator_->Free(qnn_tensor_.currentDimensions);
        qnn_tensor_.currentDimensions = nullptr;
      }

      if (qnn_tensor_.memType == QNN_TENSORMEMTYPE_RAW && nullptr != qnn_tensor_.clientBuf.data) {
        cpu_allocator_->Free(qnn_tensor_.clientBuf.data);
        qnn_tensor_.clientBuf.data = nullptr;
      }
    }
  }

  const Qnn_Tensor_t& GetQnnTensor() const { return qnn_tensor_; }
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
  onnxruntime::AllocatorPtr cpu_allocator_ = nullptr;
  Qnn_Tensor_t qnn_tensor_ = QNN_TENSOR_INIT;
};

class QnnParamWrapper {
 public:
  QnnParamWrapper(onnxruntime::AllocatorPtr cpu_allocator,
                  const std::string& name,
                  Qnn_Scalar_t scalarParam) : cpu_allocator_(cpu_allocator) {
    qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
    SetName(name);
    qnn_param_.scalarParam = scalarParam;
  }

  QnnParamWrapper(onnxruntime::AllocatorPtr cpu_allocator,
                  NodeIndex node_index,
                  const std::string& node_name,
                  const std::string& name,
                  std::vector<uint32_t>&& shape,
                  std::vector<uint32_t>&& param_data,
                  bool is_signed = false) : cpu_allocator_(cpu_allocator) {
    qnn_param_.paramType = QNN_PARAMTYPE_TENSOR;
    SetName(name);
    std::stringstream ss;
    ss << node_name << "_" << node_index << "_" << name;
    qnn_param_.tensorParam = QNN_TENSOR_INIT;
    qnn_param_.tensorParam.id = utils::GetTensorIdFromName(ss.str());
    qnn_param_.tensorParam.type = QNN_TENSOR_TYPE_STATIC;
    qnn_param_.tensorParam.dataType = is_signed ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_UINT_32;
    auto size = shape.size() * sizeof(uint32_t);
    qnn_param_.tensorParam.rank = static_cast<uint32_t>(shape.size());
    if (size > 0) {
      qnn_param_.tensorParam.memType = QNN_TENSORMEMTYPE_RAW;
      qnn_param_.tensorParam.maxDimensions = reinterpret_cast<uint32_t*>(cpu_allocator_->Alloc(size));
      qnn_param_.tensorParam.currentDimensions = reinterpret_cast<uint32_t*>(cpu_allocator_->Alloc(size));
      memscpy(qnn_param_.tensorParam.maxDimensions, size, shape.data(), size);
      memscpy(qnn_param_.tensorParam.currentDimensions, size, shape.data(), size);
      qnn_param_.tensorParam.memType = QNN_TENSORMEMTYPE_RAW;
      auto data_size = param_data.size() * sizeof(uint32_t);
      if (data_size > 0) {
        qnn_param_.tensorParam.clientBuf.data = cpu_allocator_->Alloc(data_size);
        if (is_signed) {
          int32_t* tmp = reinterpret_cast<int32_t*>(qnn_param_.tensorParam.clientBuf.data);
          std::transform(param_data.begin(), param_data.end(), tmp, [](const uint32_t& item) { return static_cast<int32_t>(item); });
        } else {
          memscpy(qnn_param_.tensorParam.clientBuf.data, data_size, param_data.data(), data_size);
        }
      } else {
        qnn_param_.tensorParam.clientBuf.data = nullptr;
      }
      qnn_param_.tensorParam.clientBuf.dataSize = static_cast<uint32_t>(data_size);
    }
    qnn_param_.tensorParam.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnParamWrapper);

  QnnParamWrapper(QnnParamWrapper&& other) noexcept {
    std::swap(cpu_allocator_, other.cpu_allocator_);
    std::swap(qnn_param_, other.qnn_param_);
  }

  ~QnnParamWrapper() {
    if (nullptr != cpu_allocator_) {
      if (nullptr != qnn_param_.name) {
        cpu_allocator_->Free(const_cast<char*>(qnn_param_.name));
        qnn_param_.name = nullptr;
      }
      if (QNN_PARAMTYPE_TENSOR == qnn_param_.paramType) {
        if (nullptr != qnn_param_.tensorParam.maxDimensions) {
          cpu_allocator_->Free(qnn_param_.tensorParam.maxDimensions);
          qnn_param_.tensorParam.maxDimensions = nullptr;
        }

        if (nullptr != qnn_param_.tensorParam.currentDimensions) {
          cpu_allocator_->Free(qnn_param_.tensorParam.currentDimensions);
          qnn_param_.tensorParam.currentDimensions = nullptr;
        }

        if (nullptr != qnn_param_.tensorParam.clientBuf.data) {
          cpu_allocator_->Free(qnn_param_.tensorParam.clientBuf.data);
          qnn_param_.tensorParam.clientBuf.data = nullptr;
        }
      }
    }
  }

  const char* GetName() const { return qnn_param_.name; }
  const Qnn_Param_t& GetQnnParam() const { return qnn_param_; }

 private:
  void SetName(const char* name) {
    if (nullptr != qnn_param_.name) {
      cpu_allocator_->Free(const_cast<char*>(qnn_param_.name));
    }
    auto size = strlen(name) + 1;
    qnn_param_.name = reinterpret_cast<char*>(cpu_allocator_->Alloc(size));
    memscpy(const_cast<char*>(qnn_param_.name), size, name, size);
  }
  void SetName(const std::string& name) {
    SetName(name.c_str());
  }

  onnxruntime::AllocatorPtr cpu_allocator_ = nullptr;
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;
};

class QnnOpConfigWrapper {
 public:
  QnnOpConfigWrapper(onnxruntime::AllocatorPtr cpu_allocator,
                     const std::string& name,
                     const std::string& package_name,
                     const std::string& type_name,
                     std::vector<Qnn_Param_t>& params,
                     std::vector<Qnn_Tensor_t>& inputs,
                     std::vector<Qnn_Tensor_t>& outputs) : cpu_allocator_(cpu_allocator) {
    SetName(name);
    SetPackageName(package_name);
    SetTypeName(type_name);
    op_config_.numOfParams = static_cast<uint32_t>(params.size());
    op_config_.numOfInputs = static_cast<uint32_t>(inputs.size());
    op_config_.numOfOutputs = static_cast<uint32_t>(outputs.size());
    if (op_config_.numOfParams > 0) {
      Qnn_Param_t* iter1 = op_config_.params = reinterpret_cast<Qnn_Param_t*>(cpu_allocator_->Alloc(params.size() * sizeof(Qnn_Param_t)));
      for (auto param : params) {
        std::swap(*iter1, param);
        ++iter1;
      }
    }
    if (op_config_.numOfInputs > 0) {
      Qnn_Tensor_t* iter2 = op_config_.inputTensors = reinterpret_cast<Qnn_Tensor_t*>(cpu_allocator_->Alloc(inputs.size() * sizeof(Qnn_Tensor_t)));
      for (auto input : inputs) {
        std::swap(*iter2, input);
        ++iter2;
      }
    }
    if (op_config_.numOfOutputs) {
      Qnn_Tensor_t* iter3 = op_config_.outputTensors = reinterpret_cast<Qnn_Tensor_t*>(cpu_allocator_->Alloc(outputs.size() * sizeof(Qnn_Tensor_t)));
      for (auto output : outputs) {
        std::swap(*iter3, output);
        ++iter3;
      }
    }
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnOpConfigWrapper);

  QnnOpConfigWrapper(QnnOpConfigWrapper&& other) noexcept {
    std::swap(cpu_allocator_, other.cpu_allocator_);
    std::swap(this->op_config_, other.op_config_);
  }

  ~QnnOpConfigWrapper() {
    if (nullptr != cpu_allocator_) {
      if (nullptr != op_config_.name) {
        cpu_allocator_->Free(const_cast<char*>(op_config_.name));
        op_config_.name = nullptr;
      }
      if (nullptr != op_config_.typeName) {
        cpu_allocator_->Free(const_cast<char*>(op_config_.typeName));
        op_config_.typeName = nullptr;
      }
      if (nullptr != op_config_.packageName) {
        cpu_allocator_->Free(const_cast<char*>(op_config_.packageName));
        op_config_.packageName = nullptr;
      }
      if (nullptr != op_config_.params) {
        cpu_allocator_->Free(op_config_.params);
        op_config_.params = nullptr;
      }
      if (nullptr != op_config_.inputTensors) {
        cpu_allocator_->Free(op_config_.inputTensors);
        op_config_.inputTensors = nullptr;
      }
      if (nullptr != op_config_.outputTensors) {
        cpu_allocator_->Free(op_config_.outputTensors);
        op_config_.outputTensors = nullptr;
      }
    }
  }

  const Qnn_OpConfig_t& GetQnnOpConfig() { return op_config_; }

 private:
  void CopyString(char** dst, const char* src) {
    if (nullptr != *dst) {
      cpu_allocator_->Free(reinterpret_cast<void*>(const_cast<char*>(op_config_.name)));
    }
    auto size = strlen(src) + 1;
    *dst = reinterpret_cast<char*>(cpu_allocator_->Alloc(size));
    memscpy(*dst, size, src, size);
  }

  void SetName(const std::string& name) {
    CopyString(const_cast<char**>(&op_config_.name), name.c_str());
  }

  void SetPackageName(const std::string& name) {
    CopyString(const_cast<char**>(&op_config_.packageName), name.c_str());
  }

  void SetTypeName(const std::string& name) {
    CopyString(const_cast<char**>(&op_config_.typeName), name.c_str());
  }

  onnxruntime::AllocatorPtr cpu_allocator_ = nullptr;
  Qnn_OpConfig_t op_config_ = QNN_OPCONFIG_INIT;
};

class GraphInfo {
 public:
  GraphInfo(Qnn_GraphHandle_t graph, const std::string& name,
            std::vector<QnnTensorWrapper>&& input_tensors,
            std::vector<QnnTensorWrapper>&& output_tensors,
            std::unordered_map<std::string, QnnTensorWrapper>&& model_tensor_map,
            std::vector<QnnParamWrapper>&& params,
            std::vector<QnnOpConfigWrapper>&& op_configs) : graph_name_(name),
                                                            graph_(graph),
                                                            input_tensors_(std::move(input_tensors)),
                                                            output_tensors_(std::move(output_tensors)),
                                                            params_(std::move(params)),
                                                            model_tensors_map_(std::move(model_tensor_map)),
                                                            op_configs_(std::move(op_configs)) {
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
  std::vector<QnnParamWrapper> params_;
  std::unordered_map<std::string, QnnTensorWrapper> model_tensors_map_;
  // Qnn_OpConfig_t are stored to avoid invalid memory access in QNN API
  // Once the requirement is cleaified and/or fixed we could avoid storing op_config data.
  std::vector<QnnOpConfigWrapper> op_configs_;
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
