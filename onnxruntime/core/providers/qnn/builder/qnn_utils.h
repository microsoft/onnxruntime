// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "QnnTypes.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <functional>
#include <numeric>
#include <vector>
#include <string>

#include "core/util/qmath.h"

namespace onnxruntime {
namespace qnn {
class QnnOpConfigWrapper;
struct OnnxInputInfo;

namespace utils {
size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type);

// TODO: make these work with Wrappers?
std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param);
std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor);
std::ostream& operator<<(std::ostream& out, const QnnOpConfigWrapper& op_conf_wrapper);

Status GetQnnDataType(const bool is_quantized_tensor, const ONNX_NAMESPACE::TypeProto* type_proto,
                      Qnn_DataType_t& tensor_data_type);

bool OnnxDataTypeToQnnDataType(const int32_t data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized = false);

inline void InitializeQuantizeParam(Qnn_QuantizeParams_t& quantize_param, bool is_quantized_tensor, float scale = 0.0f, int32_t offset = 0) {
  quantize_param.encodingDefinition = is_quantized_tensor ? QNN_DEFINITION_DEFINED : QNN_DEFINITION_UNDEFINED;
  quantize_param.quantizationEncoding = is_quantized_tensor ? QNN_QUANTIZATION_ENCODING_SCALE_OFFSET : QNN_QUANTIZATION_ENCODING_UNDEFINED;
  quantize_param.scaleOffsetEncoding.scale = scale;
  quantize_param.scaleOffsetEncoding.offset = offset;
}

// Utility function that checks if an array of strings contains a specific string.
// Used to validate ONNX operator attributes.
template <size_t N>
static bool ArrayHasString(const std::array<std::string_view, N>& strings, std::string_view str) {
  for (auto s : strings) {
    if (s == str) {
      return true;
    }
  }

  return false;
}

std::pair<float, float> CheckMinMax(float rmin, float rmax);

template <typename T>
Status GetQminQmax(const Qnn_DataType_t qnn_data_type, T& qmin, T& qmax);

template <typename T>
inline T Saturate(const T qmax,
                  const T qmin,
                  const T quant_value) {
  if (quant_value > qmax) {
    return qmax;
  } else if (quant_value < qmin) {
    return qmin;
  } else {
    return quant_value;
  }
}

Status GetQuantParams(float rmin,
                      float rmax,
                      const Qnn_DataType_t qnn_data_type,
                      float& scale,
                      int& zero_point);

double Dequantize(const OnnxInputInfo& info, const double quant_value);

Status Quantize(const double double_value,
                const float scale,
                const int zero_point,
                const Qnn_DataType_t qnn_data_type,
                int& quant_value);

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
