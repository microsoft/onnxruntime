// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "QnnInterface.h"
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
// QNN only support subset of POSIX of dlopen/dlsym/dladdr/dlerror/dlclose
// except the following flags for dlopen, others should be done only
// when we really need them
// DL_NOW is MUST
// DL_LOCAL is enabled if not specified
enum class DlOpenFlag : int {
  DL_NOW = 0x0001,
  DL_LOCAL = 0x0002,
  DL_GLOBAL = 0x0004,
};

// specify this address to distinguish from NULL pointer
#define DL_DEFAULT (void*)(0x4)

enum class ProfilingLevel : uint8_t {
  OFF = 0,
  BASIC,
  DETAILED,
  INVALID
};

// Defines performance modes available for HTP backend.
enum class HtpPerformanceMode : uint8_t {
  kHtpDefault = 0,
  kHtpSustainedHighPerformance,
  kHtpBurst,
  kHtpHighPerformance,
  kHtpPowerSaver,
  kHtpLowPowerSaver,
  kHtpHighPowerSaver,
  kHtpLowBalanced,
  kHtpBalanced,
};

enum class ContextPriority : uint8_t {
  LOW = 0,
  NORMAL,
  NORMAL_HIGH,
  HIGH,
  UNDEFINED
};

// Defines the graph optimization strategy used by the HTP backend.
enum class HtpGraphFinalizationOptimizationMode : uint8_t {
  kDefault = 0,
  kMode1 = 1,  // Faster preparation time, less optimal graph
  kMode2 = 2,  // Longer preparation time, more optimal graph
  kMode3 = 3,  // Longest preparation time, most likely even more optimal graph.
};

enum class QnnBackendType : uint8_t {
  CPU = 0,
  GPU,
  DSP,
  HTP,
  HTP_FP16
};

bool IsNpuBackend(QnnBackendType backend_type);

// constexpr config values
constexpr const int kSleepMinLatency = 40;
constexpr const int kSleepLowLatency = 100;
constexpr const int kSleepMediumLatency = 1000;
constexpr const int kSleepHighLatency = 2000;
constexpr const int kDcvsDisable = 0;
constexpr const int kDcvsEnable = 1;

struct OnnxTensorInfo {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxTensorInfo);
  OnnxTensorInfo(size_t index, int32_t data_type, std::vector<int64_t>&& shape) : index_(index), data_type_(data_type), shape_(std::move(shape)) {}
  size_t index_;
  const int32_t data_type_;  // Uses TensorProto::DataType
  const std::vector<int64_t> shape_;
};

size_t memscpy(void* dst, size_t dst_size, const void* src, size_t copy_size);

void SetQnnTensorType(Qnn_Tensor_t& qnn_tensor, Qnn_TensorType_t tensor_type);
void SetQnnTensorName(Qnn_Tensor_t& qnn_tensor, const char* name);
void SetQnnTensorDataFormat(Qnn_Tensor_t& qnn_tensor, Qnn_TensorDataFormat_t data_format);
void SetQnnTensorDataType(Qnn_Tensor_t& qnn_tensor, Qnn_DataType_t data_type);
void SetQnnTensorDim(Qnn_Tensor_t& qnn_tensor, const std::vector<uint32_t>& dimensions);
void SetQnnTensorMemType(Qnn_Tensor_t& qnn_tensor, Qnn_TensorMemType_t mem_type);
void SetQnnTensorClientBuf(Qnn_Tensor_t& qnn_tensor, const std::vector<uint8_t>& client_buf);
void SetQnnTensorClientBuf(Qnn_Tensor_t& qnn_tensor, const std::vector<uint32_t>& client_buf);
void SetQnnTensorClientBufSize(Qnn_Tensor_t& qnn_tensor, uint32_t client_buf_size);
void SetQnnTensorClientBufData(Qnn_Tensor_t& qnn_tensor, void* client_buf_data);
void SetQnnTensorQParams(Qnn_Tensor_t& qnn_tensor, const Qnn_QuantizeParams_t& quantize_params);
bool CreateTensorInQnnGraph(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                            const Qnn_GraphHandle_t& graph,
                            const std::string& node_name,
                            const std::string& tensor_name,
                            Qnn_Tensor_t& qnn_tensor,
                            std::unordered_map<std::string, bool>& tensors_created_table,
                            std::string& error_msg);

uint32_t GetQnnTensorID(const Qnn_Tensor_t& qnn_tensor);
Qnn_TensorType_t GetQnnTensorType(const Qnn_Tensor_t& qnn_tensor);
const char* GetQnnTensorName(const Qnn_Tensor_t& qnn_tensor);
Qnn_TensorDataFormat_t GetQnnTensorDataFormat(const Qnn_Tensor_t& qnn_tensor);
Qnn_DataType_t GetQnnTensorDataType(const Qnn_Tensor_t& qnn_tensor);
Qnn_TensorMemType_t GetQnnTensorMemType(const Qnn_Tensor_t& qnn_tensor);
uint32_t GetQnnTensorRank(const Qnn_Tensor_t& qnn_tensor);
uint32_t* GetQnnTensorDims(const Qnn_Tensor_t& qnn_tensor);
const Qnn_ClientBuffer_t& GetQnnTensorClientBuf(const Qnn_Tensor_t& qnn_tensor);
const Qnn_QuantizeParams_t& GetQnnTensorQParams(const Qnn_Tensor_t& qnn_tensor);

// TODO: split out separate files for Wrappers
class QnnTensorWrapper {
 public:
  QnnTensorWrapper(const std::string& name,
                   Qnn_TensorType_t tensor_type,
                   Qnn_DataType_t data_type,
                   const Qnn_QuantizeParams_t& quantize_params,
                   std::vector<uint32_t>&& shape,
                   std::vector<uint8_t>&& client_buf = {},
                   Qnn_TensorMemType_t mem_type = QNN_TENSORMEMTYPE_RAW) : tensor_name_(name),
                                                                           dimensions_(std::move(shape)),
                                                                           client_buf_(std::move(client_buf)) {
    SetQnnTensorType(qnn_tensor_, tensor_type);
    SetQnnTensorName(qnn_tensor_, tensor_name_.c_str());
    SetQnnTensorDataType(qnn_tensor_, data_type);
    SetQnnTensorDim(qnn_tensor_, dimensions_);
    SetQnnTensorMemType(qnn_tensor_, mem_type);
    if (QNN_TENSOR_TYPE_STATIC == tensor_type) {
      SetQnnTensorClientBuf(qnn_tensor_, client_buf_);
    }

    if (mem_type != QNN_TENSORMEMTYPE_RAW) {
      ORT_THROW("mem_type not supported for now.");
    }

    SetQnnTensorQParams(qnn_tensor_, quantize_params);
  }

  QnnTensorWrapper(const Qnn_Tensor_t& qnn_tensor) : tensor_name_(GetQnnTensorName(qnn_tensor)),
                                                     client_buf_{} {
    qnn_tensor_ = qnn_tensor;
    SetQnnTensorName(qnn_tensor_, tensor_name_.c_str());

    Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
    const auto& src_quantize_param = GetQnnTensorQParams(qnn_tensor);
    // quantization only support SCALE_OFFSET encoding
    quantize_param.encodingDefinition = src_quantize_param.encodingDefinition;
    quantize_param.quantizationEncoding = src_quantize_param.quantizationEncoding;
    quantize_param.scaleOffsetEncoding = src_quantize_param.scaleOffsetEncoding;
    SetQnnTensorQParams(qnn_tensor_, quantize_param);

    uint32_t shape_rank = GetQnnTensorRank(qnn_tensor);
    uint32_t* shape_data = GetQnnTensorDims(qnn_tensor);
    dimensions_.assign(shape_data, shape_data + shape_rank);
    SetQnnTensorDim(qnn_tensor_, dimensions_);

    // This method is only used for graph inputs/outputs when desearilize from cached context
    // no client buffer should be set

    SetQnnTensorMemType(qnn_tensor_, QNN_TENSORMEMTYPE_RAW);
  }

  QnnTensorWrapper() = default;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnTensorWrapper);

  QnnTensorWrapper(QnnTensorWrapper&& other) noexcept {
    std::swap(tensor_name_, other.tensor_name_);
    std::swap(dimensions_, other.dimensions_);
    std::swap(client_buf_, other.client_buf_);
    std::swap(qnn_tensor_, other.qnn_tensor_);
    SetQnnTensorName(qnn_tensor_, tensor_name_.c_str());
    SetQnnTensorDim(qnn_tensor_, dimensions_);
    SetQnnTensorClientBuf(qnn_tensor_, client_buf_);
  }

  ~QnnTensorWrapper() = default;

  const Qnn_Tensor_t& GetQnnTensor() const {
    return qnn_tensor_;
  }

  Qnn_Tensor_t& GetQnnTensor() {
    return qnn_tensor_;
  }

  const std::string& GetName() const { return tensor_name_; }

  Qnn_TensorType_t GetTensorType() const { return GetQnnTensorType(qnn_tensor_); }
  Qnn_DataType_t GetTensorDataType() const { return GetQnnTensorDataType(qnn_tensor_); }
  uint32_t GetTensorRank() const { return static_cast<uint32_t>(dimensions_.size()); }
  const std::vector<uint32_t>& GetTensorDims() const { return dimensions_; }

  bool CreateQnnGraphTensor(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                            const Qnn_GraphHandle_t& graph,
                            const std::string& node_name,
                            std::unordered_map<std::string, bool>& tensors_created_table,
                            std::string& error_msg) {
    return CreateTensorInQnnGraph(qnn_interface, graph, node_name, tensor_name_,
                                  qnn_tensor_, tensors_created_table, error_msg);
  }

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
  std::vector<uint32_t> dimensions_;
  std::vector<uint8_t> client_buf_;
  Qnn_Tensor_t qnn_tensor_ = QNN_TENSOR_INIT;
};

class QnnParamWrapper {
 public:
  QnnParamWrapper(NodeIndex node_index,
                  const std::string& node_name,
                  const std::string& name,
                  Qnn_Scalar_t scalarParam) : name_(name), shape_({}), param_data_({}) {
    qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
    qnn_param_.name = name_.c_str();
    std::stringstream ss;
    ss << node_name << "_" << node_index << "_" << name;
    tensor_name_ = ss.str();
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
    tensor_name_ = ss.str();
    qnn_param_.tensorParam = QNN_TENSOR_INIT;
    SetQnnTensorType(qnn_param_.tensorParam, QNN_TENSOR_TYPE_STATIC);
    SetQnnTensorName(qnn_param_.tensorParam, tensor_name_.c_str());
    SetQnnTensorDataType(qnn_param_.tensorParam, is_signed ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_UINT_32);
    SetQnnTensorDim(qnn_param_.tensorParam, shape_);
    SetQnnTensorMemType(qnn_param_.tensorParam, QNN_TENSORMEMTYPE_RAW);
    SetQnnTensorClientBuf(qnn_param_.tensorParam, param_data_);
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnParamWrapper);
  QnnParamWrapper(QnnParamWrapper&& other) noexcept {
    std::swap(name_, other.name_);
    std::swap(tensor_name_, other.tensor_name_);
    std::swap(shape_, other.shape_);
    std::swap(param_data_, other.param_data_);
    std::swap(qnn_param_, other.qnn_param_);
    qnn_param_.name = name_.c_str();
    if (qnn_param_.paramType == QNN_PARAMTYPE_TENSOR) {
      SetQnnTensorName(qnn_param_.tensorParam, tensor_name_.c_str());
      SetQnnTensorDim(qnn_param_.tensorParam, shape_);
      SetQnnTensorClientBuf(qnn_param_.tensorParam, param_data_);
    }
  }

  ~QnnParamWrapper() = default;

  const std::string& GetName() const {
    return name_;
  }

  const std::string& GetParamTensorName() const {
    return tensor_name_;
  }

  const Qnn_Param_t& GetQnnParam() const {
    return qnn_param_;
  }

  Qnn_Param_t& GetQnnParam() {
    return qnn_param_;
  }

  bool CreateQnnGraphParam(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                           const Qnn_GraphHandle_t& graph,
                           const std::string& node_name,
                           std::unordered_map<std::string, bool>& tensors_created_table,
                           std::string& error_msg);

 private:
  std::string name_;
  std::string tensor_name_;
  std::vector<uint32_t> shape_;
  std::vector<uint32_t> param_data_;
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;
};

class QnnOpConfigWrapper {
 public:
  QnnOpConfigWrapper(const std::string& name,
                     const std::string& package_name,
                     const std::string& type_name,
                     std::vector<Qnn_Tensor_t>&& inputs,
                     std::vector<Qnn_Tensor_t>&& outputs,
                     std::vector<Qnn_Param_t>&& params) : name_(name),
                                                          package_name_(package_name),
                                                          type_name_(type_name),
                                                          inputs_(std::move(inputs)),
                                                          outputs_(std::move(outputs)),
                                                          params_(std::move(params)) {
    SetNames(name_.c_str(), package_name_.c_str(), type_name_.c_str());
    SetNums(static_cast<uint32_t>(inputs_.size()),
            static_cast<uint32_t>(outputs_.size()),
            static_cast<uint32_t>(params_.size()));
    SetData(inputs_.data(), outputs_.data(), params_.data());
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnOpConfigWrapper);

  QnnOpConfigWrapper(QnnOpConfigWrapper&& other) noexcept {
    std::swap(this->op_config_, other.op_config_);
    std::swap(name_, other.name_);
    std::swap(package_name_, other.package_name_);
    std::swap(type_name_, other.type_name_);
    std::swap(inputs_, other.inputs_);
    std::swap(outputs_, other.outputs_);
    std::swap(params_, other.params_);
    SetNames(name_.c_str(), package_name_.c_str(), type_name_.c_str());
    SetData(inputs_.data(), outputs_.data(), params_.data());
  }

  ~QnnOpConfigWrapper() = default;

  const Qnn_OpConfig_t& GetQnnOpConfig() { return op_config_; }

  void SetNames(const char* op_name,
                const char* package_name,
                const char* type_name);
  void SetNums(uint32_t num_inputs,
               uint32_t num_outputs,
               uint32_t num_params);
  void SetData(Qnn_Tensor_t* input_tensors,
               Qnn_Tensor_t* output_tensors,
               Qnn_Param_t* params);

  const std::string& GetOpName() const { return name_; }
  const std::string& GetPackageName() const { return package_name_; }
  const std::string& GetTypeName() const { return type_name_; }
  uint32_t GetInputsNum() const { return static_cast<uint32_t>(inputs_.size()); }
  uint32_t GetOutputsNum() const { return static_cast<uint32_t>(outputs_.size()); }
  uint32_t GetParamsNum() const { return static_cast<uint32_t>(params_.size()); }
  const Qnn_Tensor_t* GetInputTensors() const { return inputs_.data(); }
  const Qnn_Tensor_t* GetOutputTensors() const { return outputs_.data(); }
  const Qnn_Param_t* GetParams() const { return params_.data(); }

  bool QnnGraphOpValidation(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                            const Qnn_BackendHandle_t& backend_handle,
                            std::string& error_msg);

  bool CreateQnnGraphOp(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                        const Qnn_GraphHandle_t& graph,
                        std::string& error_msg);

 private:
  std::string name_;
  std::string package_name_;
  std::string type_name_;
  std::vector<Qnn_Tensor_t> inputs_;
  std::vector<Qnn_Tensor_t> outputs_;
  std::vector<Qnn_Param_t> params_;
  Qnn_OpConfig_t op_config_ = QNN_OPCONFIG_INIT;
};

class QnnOpProperty {
 public:
  QnnOpProperty(const std::string& node_name,
                const std::string& package_name,
                const std::string& node_type,
                std::vector<std::string>&& input_names,
                std::vector<std::string>&& outputs_names,
                std::vector<std::string>&& param_tensor_names) : qnn_node_name_(node_name),
                                                                 package_name_(package_name),
                                                                 qnn_node_type_(node_type),
                                                                 input_names_(std::move(input_names)),
                                                                 output_names_(std::move(outputs_names)),
                                                                 param_tensor_names_(std::move(param_tensor_names)) {}

  const std::string& GetNodeName() const { return qnn_node_name_; }
  const std::string& GetPackageName() const { return package_name_; }
  const std::string& GetNodeType() const { return qnn_node_type_; }
  const std::vector<std::string>& GetInputNames() const { return input_names_; }
  const std::vector<std::string>& GetOutputNames() const { return output_names_; }
  const std::vector<std::string>& GetParamTensorNames() const { return param_tensor_names_; }

  QnnOpProperty(QnnOpProperty&& other) noexcept {
    std::swap(qnn_node_name_, other.qnn_node_name_);
    std::swap(package_name_, other.package_name_);
    std::swap(qnn_node_type_, other.qnn_node_type_);
    std::swap(input_names_, other.input_names_);
    std::swap(output_names_, other.output_names_);
    std::swap(param_tensor_names_, other.param_tensor_names_);
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnOpProperty);

 private:
  std::string qnn_node_name_;
  std::string package_name_;
  std::string qnn_node_type_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::string> param_tensor_names_;
};

class GraphInfo {
 public:
  GraphInfo(const Qnn_GraphHandle_t graph,
            const std::string& name,
            std::vector<QnnTensorWrapper>&& input_tensors,
            std::vector<QnnTensorWrapper>&& output_tensors) : graph_name_(name),
                                                              graph_(graph),
                                                              input_tensors_(std::move(input_tensors)),
                                                              output_tensors_(std::move(output_tensors)) {
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
};

typedef GraphInfo* GraphInfoPtr_t;

typedef struct GraphConfigInfo {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphConfigInfo);
  const char* graphName;
  const QnnGraph_Config_t** graphConfigs;
} GraphConfigInfo_t;

}  // namespace qnn
}  // namespace onnxruntime
