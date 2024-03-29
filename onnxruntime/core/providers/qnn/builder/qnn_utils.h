// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "QnnTypes.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <type_traits>

#include "core/util/qmath.h"

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

Status GetQnnDataType(const bool is_quantized_tensor, const ONNX_NAMESPACE::TypeProto* type_proto,
                      Qnn_DataType_t& tensor_data_type);

bool OnnxDataTypeToQnnDataType(const int32_t data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized = false);

inline void InitPerTensorQnnQuantParam(Qnn_QuantizeParams_t& quantize_param, float scale, int32_t offset = 0) {
  quantize_param.encodingDefinition = QNN_DEFINITION_DEFINED;
  quantize_param.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  quantize_param.scaleOffsetEncoding.scale = scale;
  quantize_param.scaleOffsetEncoding.offset = offset;
}

inline bool IsNotQuantized(const Qnn_QuantizeParams_t& quantize_param) {
  return quantize_param.encodingDefinition == QNN_DEFINITION_UNDEFINED;
}

inline bool IsPerTensorQuantization(const Qnn_QuantizeParams_t& quantize_param) {
  return quantize_param.encodingDefinition == QNN_DEFINITION_DEFINED &&
         (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
          quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
}

inline bool IsPerAxisQuantization(const Qnn_QuantizeParams_t& quantize_param) {
  return quantize_param.encodingDefinition == QNN_DEFINITION_DEFINED &&
         (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET ||
          quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
}

template <typename IntType>
static Status InvertPerm(gsl::span<const IntType> perm, /*out*/ gsl::span<IntType> perm_inv) {
  static_assert(std::is_integral<IntType>::value, "permutation arrays must contain integer elements");

  size_t rank = perm.size();
  ORT_RETURN_IF_NOT(perm_inv.size() == rank, "perm.size() != perm_inv.size()");

  for (size_t i = 0; i < rank; ++i) {
    size_t j = static_cast<size_t>(perm[i]);
    ORT_RETURN_IF_NOT(j < rank, "perm element out of range [0, rank - 1]");
    perm_inv[j] = static_cast<IntType>(i);
  }

  return Status::OK();
}

template <typename IntType>
static Status TryTransposeQnnQuantParams(Qnn_QuantizeParams_t& quantize_param, gsl::span<const IntType> perm) {
  if (!IsPerAxisQuantization(quantize_param)) {
    return Status::OK();
  }

  if (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    ORT_RETURN_IF_NOT(static_cast<size_t>(quantize_param.axisScaleOffsetEncoding.axis) < perm.size(),
                      "Axis value is out of range of the provided permutation");
    const int32_t new_axis = static_cast<int32_t>(perm[quantize_param.axisScaleOffsetEncoding.axis]);
    quantize_param.axisScaleOffsetEncoding.axis = new_axis;
  } else if (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    ORT_RETURN_IF_NOT(static_cast<size_t>(quantize_param.bwAxisScaleOffsetEncoding.axis) < perm.size(),
                      "Axis value is out of range of the provided permutation");
    const int32_t new_axis = static_cast<int32_t>(perm[quantize_param.bwAxisScaleOffsetEncoding.axis]);
    quantize_param.bwAxisScaleOffsetEncoding.axis = new_axis;
  }

  return Status::OK();
}

template <typename IntType>
static Status HandleUnsqueezeOnQnnQuantParams(Qnn_QuantizeParams_t& quantize_param,
                                              gsl::span<const IntType> orig_shape,
                                              gsl::span<const IntType> new_shape) {
  if (!IsPerAxisQuantization(quantize_param)) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(orig_shape.size() < new_shape.size(), "Expected unsqueezed shape to have a greater rank.");

  // Get the axis value.
  int32_t axis = 0;
  if (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    axis = quantize_param.axisScaleOffsetEncoding.axis;
  } else if (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    axis = quantize_param.bwAxisScaleOffsetEncoding.axis;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Unhandled quantization encoding: ", quantize_param.quantizationEncoding);
  }

  // Find where the axis was moved to after unsqueeze.
  size_t num_found = 0;
  size_t j = 0;
  for (size_t i = 0; i < orig_shape.size() && j < new_shape.size(); i++) {
    while (orig_shape[i] != new_shape[j] && j < new_shape.size()) {
      assert(new_shape[j] == 1);
      j++;
    }
    assert(orig_shape[i] == new_shape[j]);
    if (num_found == static_cast<size_t>(axis)) {
      break;
    }
    num_found += 1;
    j++;
  }

  if (j == static_cast<size_t>(axis)) {
    return Status::OK();
  }

  // TODO: Remove
  assert(false);  // We shouldn't run into this case yet.

  // Set new axis.
  if (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    quantize_param.axisScaleOffsetEncoding.axis = static_cast<int32_t>(j);
  } else if (quantize_param.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    quantize_param.bwAxisScaleOffsetEncoding.axis = static_cast<int32_t>(j);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Unhandled quantization encoding: ", quantize_param.quantizationEncoding);
  }

  return Status::OK();
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

double Dequantize(int32_t offset, float scale, const double quant_value);

Status Quantize(const double double_value,
                const float scale,
                const int zero_point,
                const Qnn_DataType_t qnn_data_type,
                int& quant_value);

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
