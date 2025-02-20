// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_set>

#include <gsl/gsl>

#include "nlohmann/json.hpp"
#include "QnnInterface.h"
#include "QnnTypes.h"

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
class QnnOpConfigWrapper;
class QnnModelWrapper;

namespace utils {
size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type);

// Class that allows building a JSON representation of a QNN graph.
// The JSON graph is built in a format that can be loaded with Qualcomm's QNN Netron visualizer.
class QnnJSONGraph {
 public:
  QnnJSONGraph();

  /**
   * Add QNN operator to JSON graph.
   *
   * /param op_conf_wrapper QNN operator to add.
   */
  void AddOp(const QnnOpConfigWrapper& op_conf_wrapper);

  /**
   * Finalizes JSON graph (i.e., adds top-level graph metadata) and returns a reference
   * to the JSON object.
   *
   * /return A const reference to the finalized JSON graph object.
   */
  const nlohmann::json& Finalize();

 private:
  void AddOpTensors(gsl::span<const Qnn_Tensor_t> tensors);

  nlohmann::json json_;
  std::unordered_set<std::string> seen_tensors_;   // Tracks tensors already added to JSON graph.
  std::unordered_set<std::string> seen_op_types_;  // Tracks unique operator types.
};

size_t GetElementSizeByType(ONNX_NAMESPACE::TensorProto_DataType onnx_type);

size_t GetQnnTensorDataSizeInBytes(gsl::span<const uint32_t> shape, Qnn_DataType_t element_data_type);

bool QnnTensorHasDynamicShape(const Qnn_Tensor_t& tensor);

// TODO: make these work with Wrappers?
std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param);
std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor);
std::ostream& operator<<(std::ostream& out, const QnnOpConfigWrapper& op_conf_wrapper);

Status GetQnnDataType(const bool is_quantized_tensor, const ONNX_NAMESPACE::TypeProto* type_proto,
                      Qnn_DataType_t& tensor_data_type);

const std::string& GetNodeName(const NodeUnit& node_unit);

bool OnnxDataTypeToQnnDataType(const int32_t data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized = false);

inline Status GetOnnxTensorElemDataType(const NodeArg& node_arg, /*out*/ int32_t& onnx_data_type) {
  auto type_proto = node_arg.TypeAsProto();
  ORT_RETURN_IF_NOT(type_proto != nullptr && type_proto->has_tensor_type() && type_proto->tensor_type().has_elem_type(),
                    "NodeArg must have a tensor TypeProto");
  onnx_data_type = type_proto->tensor_type().elem_type();
  return Status::OK();
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
Status GetQminQmax(const Qnn_DataType_t qnn_data_type,
                   T& qmin,
                   T& qmax,
                   bool symmetric = false) {
  if (qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8) {
    qmin = static_cast<T>(std::numeric_limits<int8_t>::min() + static_cast<int8_t>(symmetric));
    qmax = static_cast<T>(std::numeric_limits<int8_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8) {
    qmin = static_cast<T>(std::numeric_limits<uint8_t>::min());
    qmax = static_cast<T>(std::numeric_limits<uint8_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16) {
    qmin = static_cast<T>(std::numeric_limits<int16_t>::min() + static_cast<int16_t>(symmetric));
    qmax = static_cast<T>(std::numeric_limits<int16_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
    qmin = static_cast<T>(std::numeric_limits<uint16_t>::min());
    qmax = static_cast<T>(std::numeric_limits<uint16_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32) {
    qmin = static_cast<T>(std::numeric_limits<int32_t>::min() + static_cast<int32_t>(symmetric));
    qmax = static_cast<T>(std::numeric_limits<int32_t>::max());
  } else {
    ORT_RETURN_IF(true, "Qnn Data Type: %d not supported yet.", qnn_data_type);
  }
  return Status::OK();
}

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
                      int32_t& zero_point,
                      bool symmetric = false);

double Dequantize(int32_t offset, float scale, const double quant_value);

Status Quantize(const double double_value,
                const float scale,
                const int32_t zero_point,
                const Qnn_DataType_t qnn_data_type,
                int& quant_value);

size_t ShapeSizeCalc(gsl::span<const uint32_t> shape, size_t start, size_t end);

// Computes the quantization parameters (scales and offsets) for the given data.
// Supports both per-tensor and per-channel quantization. Must provide an axis argument
// for per-channel quantization.
// The offsets use the QNN convention where offset = -zero_point.
Status GetDataQuantParams(gsl::span<const float> data, gsl::span<const uint32_t> shape,
                          /*out*/ gsl::span<float> scales, /*out*/ gsl::span<int32_t> offsets,
                          Qnn_DataType_t data_type, bool symmetric = false,
                          std::optional<int64_t> axis = std::nullopt);

// Quantizes the given float data using the provided quantization parameters (scales and offsets).
// Supports both per-tensor and per-channel quantization. Must provide an axis argument
// for per-channel quantization.
// The provided offsets must use the QNN convention where offset = -zero_point.
Status QuantizeData(gsl::span<const float> data, gsl::span<const uint32_t> shape,
                    gsl::span<const float> scales, gsl::span<const int32_t> offsets,
                    /*out*/ gsl::span<uint8_t> quant_bytes, Qnn_DataType_t data_type,
                    std::optional<int64_t> axis = std::nullopt);

// Quantizes (per-tensor) the given float data using the provided scale and offset.
// The provided offset must use the QNN convention where offset = -zero_point.
template <typename QuantType>
inline Status QuantizeData(gsl::span<const float> data, float scale, int32_t offset,
                           /*out*/ gsl::span<uint8_t> quant_bytes) {
  const size_t num_elems = data.size();
  const size_t expected_output_bytes = sizeof(QuantType) * num_elems;
  ORT_RETURN_IF_NOT(expected_output_bytes == quant_bytes.size(),
                    "Output buffer is not large enough to hold quantized bytes.");
  const double clip_min = static_cast<double>(std::numeric_limits<QuantType>::lowest());
  const double clip_max = static_cast<double>(std::numeric_limits<QuantType>::max());

  QuantType* output = reinterpret_cast<QuantType*>(quant_bytes.data());
  for (size_t i = 0; i < num_elems; ++i) {
    const double scale_dbl = static_cast<double>(scale);
    const double offset_dbl = static_cast<double>(offset);
    double float_val = std::nearbyint(static_cast<double>(data[i]) / scale_dbl) - offset_dbl;
    float_val = std::max(float_val, clip_min);
    float_val = std::min(float_val, clip_max);
    output[i] = static_cast<QuantType>(float_val);
  }
  return Status::OK();
}

// Re-writes a buffer of packed 4-bit elements to a buffer of unpacked 8-bit elements.
// QNN requires that 4-bit weights are unpacked to 8-bit.
template <bool Signed>
Status UnpackInt4ToInt8(size_t num_int4_elems, std::vector<uint8_t>& data_bytes) {
  if constexpr (Signed) {  // INT4
    std::vector<uint8_t> packed_int4_bytes = std::move(data_bytes);
    data_bytes = std::vector<uint8_t>(num_int4_elems);

    auto dst = gsl::make_span(reinterpret_cast<int8_t*>(data_bytes.data()), data_bytes.size());
    auto src = gsl::make_span(reinterpret_cast<const Int4x2*>(packed_int4_bytes.data()), packed_int4_bytes.size());
    ORT_RETURN_IF_NOT(Int4x2::Unpack(dst, src), "Failed to unpack Tensor<Int4x2> for QNN");

    // NOTE: Masking off top 4 bits to workaround a QNN INT4 accuracy bug.
    // Docs explicitly state that masking off top 4 bits should not be required, but we have to do it.
    for (size_t i = 0; i < dst.size(); i++) {
      dst[i] &= 0x0F;  // -3 (0b1111_1101) becomes 13 (0b0000_1101)
    }
  } else {  // UINT4
    std::vector<uint8_t> packed_uint4_bytes = std::move(data_bytes);
    data_bytes = std::vector<uint8_t>(num_int4_elems);

    auto dst = gsl::make_span(reinterpret_cast<uint8_t*>(data_bytes.data()), data_bytes.size());
    auto src = gsl::make_span(reinterpret_cast<const UInt4x2*>(packed_uint4_bytes.data()), packed_uint4_bytes.size());
    ORT_RETURN_IF_NOT(UInt4x2::Unpack(dst, src), "Failed to unpack Tensor<UInt4x2> for QNN");
  }

  return Status::OK();
}

template <typename T>
std::vector<T> GetInitializerShape(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<T> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = static_cast<T>(dims[i]);
  }

  return tensor_shape_vec;
}

TensorShape GetTensorProtoShape(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto);

template <typename T, typename P>
Status PermuteShape(gsl::span<const T> input_shape, gsl::span<const P> perm, gsl::span<T> output_shape) {
  const size_t rank = input_shape.size();
  ORT_RETURN_IF_NOT(rank == perm.size() && rank == output_shape.size(),
                    "PermuteShape(): expect all arguments to have the same rank.");

  for (size_t i = 0; i < rank; ++i) {
    size_t p = static_cast<size_t>(perm[i]);
    output_shape[i] = input_shape[p];
  }

  return Status::OK();
}

// Gets error message associated with QNN error handle value.
std::string GetQnnErrorMessage(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                               Qnn_ErrorHandle_t qnn_error_handle);

// Gets verbose error message associated with QNN error handle value.
std::string GetVerboseQnnErrorMessage(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                      Qnn_ErrorHandle_t qnn_error_handle);

// NCHW shape to channel last
template <typename T>
Status NchwShapeToNhwc(gsl::span<const T> nchw_shape, gsl::span<T> nhwc_shape) {
  ORT_RETURN_IF_NOT(nchw_shape.size() == 4, "shape should have 4 dimension NCHW.");
  nhwc_shape[0] = nchw_shape[0];
  nhwc_shape[1] = nchw_shape[2];
  nhwc_shape[2] = nchw_shape[3];
  nhwc_shape[3] = nchw_shape[1];

  return Status::OK();
}

// NCHW shape to HWCN shape, required for Conv weight
template <typename T>
Status NchwShapeToHwcn(gsl::span<const T> nchw_shape, gsl::span<T> hwcn_shape) {
  if (nchw_shape.size() == 4) {
    hwcn_shape[0] = nchw_shape[2];
    hwcn_shape[1] = nchw_shape[3];
    hwcn_shape[2] = nchw_shape[1];
    hwcn_shape[3] = nchw_shape[0];
  } else if (nchw_shape.size() == 5) {
    hwcn_shape[0] = nchw_shape[2];
    hwcn_shape[1] = nchw_shape[3];
    hwcn_shape[2] = nchw_shape[4];
    hwcn_shape[3] = nchw_shape[1];
    hwcn_shape[4] = nchw_shape[0];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported rank! only support 4 or 5.");
  }

  return Status::OK();
}

// CNHW shape to HWCN shape, required for Conv weight
template <typename T>
Status CnhwShapeToHwcn(gsl::span<const T> cnhw_shape, gsl::span<T> hwcn_shape) {
  if (cnhw_shape.size() == 4) {
    hwcn_shape[0] = cnhw_shape[2];
    hwcn_shape[1] = cnhw_shape[3];
    hwcn_shape[2] = cnhw_shape[0];
    hwcn_shape[3] = cnhw_shape[1];
  } else if (cnhw_shape.size() == 5) {
    hwcn_shape[0] = cnhw_shape[2];
    hwcn_shape[1] = cnhw_shape[3];
    hwcn_shape[2] = cnhw_shape[4];
    hwcn_shape[3] = cnhw_shape[0];
    hwcn_shape[4] = cnhw_shape[1];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported rank! only support 4 or 5.");
  }

  return Status::OK();
}

Status TransposeFromNchwToHwcn(const QnnModelWrapper& qnn_model_wrapper,
                               const onnx::TensorProto& initializer,
                               std::vector<uint8_t>& transposed_data,
                               bool is_3d = false);
Status TransposeFromNchwToHwcn(std::vector<int64_t>&& input_shape_dims,
                               size_t elem_byte_size,
                               gsl::span<const uint8_t> input_buffer,
                               gsl::span<uint8_t> output_buffer,
                               bool is_3d = false);

Status TransposeFromCnhwToHwcn(const QnnModelWrapper& qnn_model_wrapper,
                               const onnx::TensorProto& initializer,
                               std::vector<uint8_t>& transposed_data,
                               bool is_3d = false);
Status TransposeFromCnhwToHwcn(std::vector<int64_t>&& input_shape_dims,
                               size_t elem_byte_size,
                               gsl::span<const uint8_t> input_buffer,
                               gsl::span<uint8_t> output_buffer,
                               bool is_3d = false);

Status TwoDimensionTranspose(const QnnModelWrapper& qnn_model_wrapper,
                             std::vector<uint32_t>& data_shape,
                             const onnx::TensorProto& initializer,
                             std::vector<uint8_t>& transposed_data);

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
