// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "QnnTypes.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <functional>
#include <numeric>
#include <vector>
#include <string>

namespace onnxruntime {
namespace qnn {
class QnnOpConfigWrapper;

namespace utils {
size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type);

// TODO: make these work with Wrappers?
std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param);
std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor);
std::ostream& operator<<(std::ostream& out, const QnnOpConfigWrapper& op_conf_wrapper);

Status GetQnnDataType(const bool is_quantized_node, const ONNX_NAMESPACE::TypeProto* type_proto,
                      Qnn_DataType_t& tensor_data_type);

bool OnnxDataTypeToQnnDataType(const int32_t data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized = false);

inline void InitializeQuantizeParam(Qnn_QuantizeParams_t& quantize_param, bool is_quantized_model, float scale = 0.0f, int32_t offset = 0) {
  quantize_param.encodingDefinition = is_quantized_model ? QNN_DEFINITION_DEFINED : QNN_DEFINITION_UNDEFINED;
  quantize_param.quantizationEncoding = is_quantized_model ? QNN_QUANTIZATION_ENCODING_SCALE_OFFSET : QNN_QUANTIZATION_ENCODING_UNDEFINED;
  quantize_param.scaleOffsetEncoding.scale = scale;
  quantize_param.scaleOffsetEncoding.offset = offset;
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
