// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
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

#include "core/providers/qnn-abi/ort_api.h"
// Forward declaration to avoid circular dependency

namespace onnxruntime {
namespace qnn {
class QnnOpConfigWrapper;
class QnnModelWrapper;

namespace utils {
/**
 * Returns a lowercase version of the input string.
 * /param str The string to lowercase.
 * /return The lowercased string.
 */
inline std::string GetLowercaseString(std::string str) {
  // https://en.cppreference.com/w/cpp/string/byte/tolower
  // The behavior of tolower from <cctype> is undefined if the argument is neither representable as unsigned char
  // nor equal to EOF. To use tolower safely with a plain char (or signed char), the argument must be converted to
  // unsigned char.
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return str;
}

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

Status GetQnnDataType(const bool is_quantized_tensor,
                      const ONNXTensorElementDataType onnx_data_type,
                      Qnn_DataType_t& tensor_data_type);

const std::string& GetNodeName(const OrtNodeUnit& node_unit);

bool OnnxDataTypeToQnnDataType(const ONNXTensorElementDataType onnx_data_type,
                               Qnn_DataType_t& qnn_data_type,
                               bool is_quantized = false);

// inline Status GetOnnxTensorElemDataType(const NodeArg& node_arg, /*out*/ int32_t& onnx_data_type) {
//   auto type_proto = node_arg.TypeAsProto();
//   ORT_RETURN_IF_NOT(type_proto != nullptr && type_proto->has_tensor_type() && type_proto->tensor_type().has_elem_type(),
//                     "NodeArg must have a tensor TypeProto");
//   onnx_data_type = type_proto->tensor_type().elem_type();
//   return Status::OK();
// }

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

inline std::vector<int64_t> GetInitializerShape(const OrtValueInfo& initializer, const OrtApi& ort_api) {
  const OrtTypeInfo* type_info = nullptr;
  OrtStatus* status = ort_api.GetValueInfoTypeInfo(static_cast<const OrtValueInfo*>(&initializer), &type_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return {};
  }

  const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info = nullptr;
  status = ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_type_and_shape_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
    return {};
  }

  size_t num_dims;
  status = ort_api.GetDimensionsCount(tensor_type_and_shape_info, &num_dims);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
    ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info));
    return {};
  }

  std::vector<int64_t> dims;
  dims.resize(num_dims, 0);
  status = ort_api.GetDimensions(tensor_type_and_shape_info, dims.data(), dims.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
    ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info));
    return {};
  }

  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (size_t i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = static_cast<int64_t>(dims[i]);
  }

  ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
  ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info));
  return tensor_shape_vec;
}

// TensorShape GetTensorProtoShape(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto);

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

// // Gets verbose error message associated with QNN error handle value.
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
                               OrtValueInfo& initializer,
                               std::vector<uint8_t>& transposed_data,
                               bool is_3d = false);
Status TransposeFromNchwToHwcn(std::vector<int64_t>&& input_shape_dims,
                               size_t elem_byte_size,
                               gsl::span<const uint8_t> input_buffer,
                               gsl::span<uint8_t> output_buffer,
                               bool is_3d = false);

Status TransposeFromCnhwToHwcn(const QnnModelWrapper& qnn_model_wrapper,
                               OrtValueInfo& initializer,
                               std::vector<uint8_t>& transposed_data,
                               bool is_3d = false);
Status TransposeFromCnhwToHwcn(std::vector<int64_t>&& input_shape_dims,
                               size_t elem_byte_size,
                               gsl::span<const uint8_t> input_buffer,
                               gsl::span<uint8_t> output_buffer,
                               bool is_3d = false);

Status TwoDimensionTranspose(const QnnModelWrapper& qnn_model_wrapper,
                             std::vector<uint32_t>& data_shape,
                             OrtValueInfo& initializer,
                             std::vector<uint8_t>& transposed_data);

Status InsertConvertOp(QnnModelWrapper& qnn_model_wrapper,
                       const std::string& convert_input_name,
                       const std::string& convert_output_name,
                       Qnn_DataType_t input_qnn_data_type,
                       Qnn_DataType_t output_qnn_data_type,
                       int32_t input_offset,
                       float input_scale,
                       const std::vector<uint32_t>& output_shape,
                       bool output_symmetric,
                       bool do_op_validation);

/**
 * Get permutation to transpose given axis to the last one.
 *
 * @param[in] axis the current axis to be transposed
 * @param[in] rank the expected rank for permutation
 * @param[out] perm the permutation for transpose
 * @return execution status of this function
 */
Status GetPermToLastAxis(uint32_t axis, uint32_t rank, std::vector<uint32_t>& perm);

#define CASE_UNPACK(TYPE, ELEMENT_TYPE, DATA_SIZE)                                                             \
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_##TYPE: {                                      \
    const OrtTypeInfo* type_info_##TYPE = nullptr;                                                             \
    status = ort_api.GetValueInfoTypeInfo(static_cast<const OrtValueInfo*>(&initializer), &type_info_##TYPE); \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get value info type info");                  \
    }                                                                                                          \
    const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info_##TYPE = nullptr;                              \
    status = ort_api.CastTypeInfoToTensorInfo(type_info_##TYPE, &tensor_type_and_shape_info_##TYPE);           \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to cast type info to tensor info");             \
    }                                                                                                          \
    size_t num_dims_##TYPE = 0;                                                                                \
    status = ort_api.GetDimensionsCount(tensor_type_and_shape_info_##TYPE, &num_dims_##TYPE);                  \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get dimensions count");                      \
    }                                                                                                          \
    std::vector<int64_t> dims_##TYPE(num_dims_##TYPE);                                                         \
    status = ort_api.GetDimensions(tensor_type_and_shape_info_##TYPE, dims_##TYPE.data(), dims_##TYPE.size()); \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get dimensions");                            \
    }                                                                                                          \
                                                                                                               \
    /* Calculate element count from shape */                                                                   \
    size_t element_count = 1;                                                                                  \
    for (size_t i = 0; i < num_dims_##TYPE; ++i) {                                                             \
      element_count *= static_cast<size_t>(dims_##TYPE[i]);                                                    \
    }                                                                                                          \
                                                                                                               \
    /* Calculate tensor byte size */                                                                           \
    size_t tensor_byte_size = element_count * sizeof(ELEMENT_TYPE);                                            \
                                                                                                               \
    /* Get tensor data */                                                                                      \
    const OrtValue* initializer_value = nullptr;                                                               \
    status = ort_api.ValueInfo_GetInitializerValue(static_cast<const OrtValueInfo*>(&initializer), &initializer_value); \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get initializer value");                     \
    }                                                                                                          \
    const void* data = nullptr;                                                                                \
    status = ort_api.GetTensorData(initializer_value, &data);                                                  \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get tensor data");                           \
    }                                                                                                          \
                                                                                                               \
    /* Resize output buffer and copy data */                                                                   \
    unpacked_tensor.resize(tensor_byte_size);                                                                  \
    if (data != nullptr && tensor_byte_size > 0) {                                                             \
      std::memcpy(unpacked_tensor.data(), data, tensor_byte_size);                                             \
    }                                                                                                          \
                                                                                                               \
    ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                       \
    ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
    return Status::OK();                                                                                       \
  }

#define CASE_UNPACK_INT4(TYPE, ELEMENT_TYPE, DATA_SIZE)                                                        \
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_##TYPE: {                                      \
    /* Get tensor shape to calculate element count */                                                          \
    const OrtTypeInfo* type_info_##TYPE = nullptr;                                                             \
    status = ort_api.GetValueInfoTypeInfo(static_cast<const OrtValueInfo*>(&initializer), &type_info_##TYPE); \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get value info type info");                  \
    }                                                                                                          \
    const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info_##TYPE = nullptr;                              \
    status = ort_api.CastTypeInfoToTensorInfo(type_info_##TYPE, &tensor_type_and_shape_info_##TYPE);           \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to cast type info to tensor info");             \
    }                                                                                                          \
    size_t num_dims_##TYPE = 0;                                                                                \
    status = ort_api.GetDimensionsCount(tensor_type_and_shape_info_##TYPE, &num_dims_##TYPE);                  \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get dimensions count");                      \
    }                                                                                                          \
    std::vector<int64_t> dims_##TYPE(num_dims_##TYPE);                                                         \
    status = ort_api.GetDimensions(tensor_type_and_shape_info_##TYPE, dims_##TYPE.data(), dims_##TYPE.size()); \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get dimensions");                            \
    }                                                                                                          \
                                                                                                               \
    /* Calculate element count from shape */                                                                   \
    size_t element_count = 1;                                                                                  \
    for (size_t i = 0; i < num_dims_##TYPE; ++i) {                                                             \
      element_count *= static_cast<size_t>(dims_##TYPE[i]);                                                    \
    }                                                                                                          \
                                                                                                               \
    /* Calculate packed element count and tensor byte size for INT4/UINT4 */                                   \
    size_t packed_element_count = ELEMENT_TYPE::CalcNumInt4Pairs(element_count);                               \
    size_t tensor_byte_size = packed_element_count * sizeof(ELEMENT_TYPE);                                     \
                                                                                                               \
    /* Get tensor data */                                                                                      \
    const OrtValue* initializer_value = nullptr;                                                               \
    status = ort_api.ValueInfo_GetInitializerValue(static_cast<const OrtValueInfo*>(&initializer), &initializer_value); \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get initializer value");                     \
    }                                                                                                          \
    const void* data = nullptr;                                                                                \
    status = ort_api.GetTensorData(initializer_value, &data);                                                  \
    if (status != nullptr) {                                                                                   \
      ort_api.ReleaseStatus(status);                                                                           \
      ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                     \
      ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get tensor data");                           \
    }                                                                                                          \
                                                                                                               \
    /* Resize output buffer and copy data */                                                                   \
    unpacked_tensor.resize(tensor_byte_size);                                                                  \
    if (data != nullptr && tensor_byte_size > 0) {                                                             \
      std::memcpy(unpacked_tensor.data(), data, tensor_byte_size);                                             \
    }                                                                                                          \
                                                                                                               \
    ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info_##TYPE));                                       \
    ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info_##TYPE)); \
    return Status::OK();                                                                                       \
  }

inline Status UnpackInitializerData(const OrtApi& ort_api,
                                    OrtValueInfo& initializer,
                                    const std::filesystem::path& model_path,
                                    std::vector<uint8_t>& unpacked_tensor) {
  std::cout << "DEBUG: model_path: " << model_path << std::endl;
  // TODO: Handle external data if needed
  // // TODO, if std::vector does not use a custom allocator, the default std::allocator will
  // // allocation the memory aligned to std::max_align_t, need look into allocating
  // // forced aligned memory (align as 16 or larger)for unpacked_tensor
  // if (HasExternalData(initializer)) {
  //   ORT_RETURN_IF_ERROR(ReadExternalDataForTensor(
  //       initializer,
  //       model_path.parent_path(),
  //       unpacked_tensor));
  //   return Status::OK();
  // }

  const OrtTypeInfo* type_info = nullptr;
  OrtStatus* status = ort_api.GetValueInfoTypeInfo(static_cast<const OrtValueInfo*>(&initializer), &type_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get value info type info");
  }

  const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info = nullptr;
  status = ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_type_and_shape_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to cast type info to tensor info");
  }

  ONNXTensorElementDataType onnx_data_type;
  status = ort_api.GetTensorElementType(tensor_type_and_shape_info, &onnx_data_type);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
    ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(tensor_type_and_shape_info));
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get tensor element type");
  }

  switch (onnx_data_type) {
    CASE_UNPACK(FLOAT, float, float_data_size);
    CASE_UNPACK(DOUBLE, double, double_data_size);
    CASE_UNPACK(BOOL, bool, int32_data_size);
    CASE_UNPACK(INT8, int8_t, int32_data_size);
    CASE_UNPACK(INT16, int16_t, int32_data_size);
    CASE_UNPACK(INT32, int32_t, int32_data_size);
    CASE_UNPACK(INT64, int64_t, int64_data_size);
    CASE_UNPACK(UINT8, uint8_t, int32_data_size);
    CASE_UNPACK(UINT16, uint16_t, int32_data_size);
    CASE_UNPACK(UINT32, uint32_t, uint64_data_size);
    CASE_UNPACK(UINT64, uint64_t, uint64_data_size);
    CASE_UNPACK(FLOAT16, onnxruntime::MLFloat16, int32_data_size);
    CASE_UNPACK(BFLOAT16, onnxruntime::BFloat16, int32_data_size);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_UNPACK(FLOAT8E4M3FN, onnxruntime::Float8E4M3FN, int32_data_size);
    CASE_UNPACK(FLOAT8E4M3FNUZ, onnxruntime::Float8E4M3FNUZ, int32_data_size);
    CASE_UNPACK(FLOAT8E5M2, onnxruntime::Float8E5M2, int32_data_size);
    CASE_UNPACK(FLOAT8E5M2FNUZ, onnxruntime::Float8E5M2FNUZ, int32_data_size);
#endif
    CASE_UNPACK_INT4(INT4, Int4x2, int32_data_size);
    CASE_UNPACK_INT4(UINT4, UInt4x2, int32_data_size);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported type: ", onnx_data_type);
  }
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
