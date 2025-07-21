// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cassert>
#include <optional>
#include <vector>
#include "QnnTypes.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_quant_params_wrapper.h"

#define ALIGN_PTR_UP(ptr, align, type) \
  reinterpret_cast<type>((reinterpret_cast<std::uintptr_t>(ptr) + (align) - 1) & ~((align) - 1))

namespace onnxruntime {
namespace qnn {

QnnQuantParamsWrapper::QnnQuantParamsWrapper(const QnnQuantParamsWrapper& other)
    : params_(QNN_QUANTIZE_PARAMS_INIT) {
  Status status = Init(other.params_);
  assert(status.IsOK());  // Expect other QnnQuantParamsWrapper to always have a supported quantization encoding.
}

QnnQuantParamsWrapper& QnnQuantParamsWrapper::operator=(const QnnQuantParamsWrapper& other) {
  if (this != &other) {
    Status status = Init(other.params_);
    assert(status.IsOK());  // Expect other QnnQuantParamsWrapper to always have a supported quantization encoding.
  }

  return *this;
}

// Construct per-tensor quantization params.
QnnQuantParamsWrapper::QnnQuantParamsWrapper(float scale, int32_t offset) {
  params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  params_.scaleOffsetEncoding.scale = scale;
  params_.scaleOffsetEncoding.offset = offset;
}

// Construct a per-channel quantization param.
QnnQuantParamsWrapper::QnnQuantParamsWrapper(gsl::span<const float> scales, gsl::span<const int32_t> offsets,
                                             int32_t axis, bool is_int4) {
  assert(scales.size() == offsets.size());  // Logic error if sizes don't match.
  const uint32_t num_elems = static_cast<uint32_t>(scales.size());
  params_.encodingDefinition = QNN_DEFINITION_DEFINED;

  if (is_int4) {
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
    params_.bwAxisScaleOffsetEncoding.numElements = num_elems;
    params_.bwAxisScaleOffsetEncoding.axis = axis;
    params_.bwAxisScaleOffsetEncoding.bitwidth = 4;

    // Deep copy to the scales[] and offsets[] arrays
    if (num_elems > 0) {
      const size_t num_scale_bytes = num_elems * sizeof(float);
      const size_t num_zp_bytes = num_elems * sizeof(int32_t);
      const size_t num_bytes = num_scale_bytes + num_zp_bytes;
      constexpr std::uintptr_t align = alignof(float);
      static_assert(alignof(float) == alignof(int32_t));

      per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
      char* scales_begin = ALIGN_PTR_UP(per_channel_data_.get(), align, char*);
      char* zps_begin = scales_begin + num_scale_bytes;

      std::memcpy(scales_begin, scales.data(), num_scale_bytes);
      std::memcpy(zps_begin, offsets.data(), num_zp_bytes);
      params_.bwAxisScaleOffsetEncoding.scales = reinterpret_cast<float*>(scales_begin);
      params_.bwAxisScaleOffsetEncoding.offsets = reinterpret_cast<int32_t*>(zps_begin);
    } else {
      params_.bwAxisScaleOffsetEncoding.scales = nullptr;
      params_.bwAxisScaleOffsetEncoding.offsets = nullptr;
    }
  } else {
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
    params_.axisScaleOffsetEncoding.numScaleOffsets = num_elems;
    params_.axisScaleOffsetEncoding.axis = axis;

    // Deep copy to the scaleOffset data.
    if (num_elems > 0) {
      const size_t num_bytes = num_elems * sizeof(Qnn_ScaleOffset_t);
      constexpr std::uintptr_t align = alignof(Qnn_ScaleOffset_t);
      per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
      Qnn_ScaleOffset_t* aligned_dst = ALIGN_PTR_UP(per_channel_data_.get(), align, Qnn_ScaleOffset_t*);

      for (size_t i = 0; i < static_cast<uint32_t>(num_elems); i++) {
        aligned_dst[i].offset = offsets[i];
        aligned_dst[i].scale = scales[i];
      }

      params_.axisScaleOffsetEncoding.scaleOffset = aligned_dst;
    } else {
      params_.axisScaleOffsetEncoding.scaleOffset = nullptr;
    }
  }
}

// Get a copy of scales. Works for both per-tensor and per-channel.
Status QnnQuantParamsWrapper::GetScales(/*out*/ std::vector<float>& scales) const {
  ORT_RETURN_IF_NOT(params_.encodingDefinition == QNN_DEFINITION_DEFINED, "Unquantized qparams does not have scales");

  switch (params_.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      scales.resize(1);
      scales[0] = params_.scaleOffsetEncoding.scale;
      break;
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      scales.resize(1);
      scales[0] = params_.bwScaleOffsetEncoding.scale;
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      const uint32_t num_elems = params_.axisScaleOffsetEncoding.numScaleOffsets;
      scales.resize(num_elems);

      if (num_elems > 0) {
        gsl::span<const Qnn_ScaleOffset_t> scale_offsets(params_.axisScaleOffsetEncoding.scaleOffset, num_elems);

        for (size_t i = 0; i < num_elems; i++) {
          scales[i] = scale_offsets[i].scale;
        }
      }
      break;
    }
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
      const uint32_t num_elems = params_.bwAxisScaleOffsetEncoding.numElements;
      scales.resize(num_elems);

      // Deep copy the scales[] and offsets[] arrays
      if (num_elems > 0) {
        gsl::span<const float> src_scales(params_.bwAxisScaleOffsetEncoding.scales, num_elems);
        for (size_t i = 0; i < num_elems; i++) {
          scales[i] = src_scales[i];
        }
      }
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported QNN quantization encoding: ",
                             params_.quantizationEncoding);
  }

  return Status::OK();
}

QnnQuantParamsWrapper QnnQuantParamsWrapper::Copy() const {
  return QnnQuantParamsWrapper(*this);
}

// Initializes by copying from a Qnn_QuantizeParams_t.
Status QnnQuantParamsWrapper::Init(const Qnn_QuantizeParams_t& params) {
  if (per_channel_data_) {
    per_channel_data_.reset(nullptr);
    params_ = QNN_QUANTIZE_PARAMS_INIT;
  }

  if (params.encodingDefinition != QNN_DEFINITION_DEFINED) {
    params_ = params;
    return Status::OK();
  }

  switch (params.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      params_ = params;
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      params_.encodingDefinition = params.encodingDefinition;
      params_.quantizationEncoding = params.quantizationEncoding;
      params_.axisScaleOffsetEncoding.axis = params.axisScaleOffsetEncoding.axis;
      params_.axisScaleOffsetEncoding.numScaleOffsets = params.axisScaleOffsetEncoding.numScaleOffsets;

      // Deep copy the scaleOffset data.
      const uint32_t num_elems = params.axisScaleOffsetEncoding.numScaleOffsets;

      if (num_elems > 0) {
        const size_t num_bytes = num_elems * sizeof(Qnn_ScaleOffset_t);
        constexpr std::uintptr_t align = alignof(Qnn_ScaleOffset_t);
        per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
        Qnn_ScaleOffset_t* aligned_dst = ALIGN_PTR_UP(per_channel_data_.get(), align, Qnn_ScaleOffset_t*);

        std::memcpy(aligned_dst, params.axisScaleOffsetEncoding.scaleOffset, num_bytes);
        params_.axisScaleOffsetEncoding.scaleOffset = aligned_dst;
      } else {
        params_.axisScaleOffsetEncoding.scaleOffset = nullptr;
      }
      break;
    }
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
      const uint32_t num_elems = params.bwAxisScaleOffsetEncoding.numElements;

      params_.encodingDefinition = params.encodingDefinition;
      params_.quantizationEncoding = params.quantizationEncoding;
      params_.bwAxisScaleOffsetEncoding.axis = params.bwAxisScaleOffsetEncoding.axis;
      params_.bwAxisScaleOffsetEncoding.bitwidth = params.bwAxisScaleOffsetEncoding.bitwidth;
      params_.bwAxisScaleOffsetEncoding.numElements = num_elems;

      // Deep copy the scales[] and offsets[] arrays
      if (num_elems > 0) {
        const size_t num_scale_bytes = num_elems * sizeof(float);
        const size_t num_zp_bytes = num_elems * sizeof(int32_t);
        const size_t num_bytes = num_scale_bytes + num_zp_bytes;
        constexpr std::uintptr_t align = alignof(float);
        static_assert(alignof(float) == alignof(int32_t));

        per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
        char* scales_begin = ALIGN_PTR_UP(per_channel_data_.get(), align, char*);
        char* zps_begin = scales_begin + num_scale_bytes;

        std::memcpy(scales_begin, params.bwAxisScaleOffsetEncoding.scales, num_scale_bytes);
        std::memcpy(zps_begin, params.bwAxisScaleOffsetEncoding.offsets, num_zp_bytes);
        params_.bwAxisScaleOffsetEncoding.scales = reinterpret_cast<float*>(scales_begin);
        params_.bwAxisScaleOffsetEncoding.offsets = reinterpret_cast<int32_t*>(zps_begin);
      } else {
        params_.bwAxisScaleOffsetEncoding.scales = nullptr;
        params_.bwAxisScaleOffsetEncoding.offsets = nullptr;
      }
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported QNN quantization encoding: ", params.quantizationEncoding);
  }

  return Status::OK();
}

// Initialize this object from a (potentially) quantized ONNX tensor.
// QnnModelWrapper provides utilities for unpacking scale and zero-point ONNX initializers.
Status QnnQuantParamsWrapper::Init(const OrtApi& ort_api,
                                   const QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnitIODef& io_def) {
  const std::optional<OrtNodeUnitIODef::QuantParam>& ort_quant_params = io_def.quant_param;

  if (per_channel_data_) {
    per_channel_data_.reset(nullptr);
    params_ = QNN_QUANTIZE_PARAMS_INIT;
  }

  if (!ort_quant_params.has_value()) {
    params_.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    return Status::OK();
  }

  std::vector<float> scales;
  std::vector<int32_t> zero_points;

  // TODO: Check the type of io_def.quant_param->scale
  // According to the type definition, it may need to be revised.
  const OrtValueInfo* qparam_scale = static_cast<const OrtValueInfo*>(ort_quant_params->scale);
  const char* qparam_scale_name = nullptr;
  ort_api.GetValueInfoName(qparam_scale, &qparam_scale_name);
  const std::string& scale_name = std::string(qparam_scale_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(scale_name, scales));

  bool is_int4_type = false;

  if (ort_quant_params->zero_point != nullptr) {
    // TODO: Check the type of io_def.quant_param->zero_point
    // According to the type definition, it may need to be revised.
    const OrtValueInfo* qparam_zero_point = static_cast<const OrtValueInfo*>(ort_quant_params->zero_point);
    const char* qparam_zero_point_name = nullptr;
    ort_api.GetValueInfoName(qparam_zero_point, &qparam_zero_point_name);
    const std::string& zero_point_name = std::string(qparam_zero_point_name);

    ONNXTensorElementDataType onnx_tp_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(zero_point_name, zero_points,
                                                           onnx_tp_type));

    is_int4_type = (onnx_tp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) ||
                   (onnx_tp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
  }

  const bool is_per_tensor = scales.size() == 1;

  // QNN uses different structs to represent quantization parameters depending on
  // - per-tensor vs per-channel
  // - int4 vs not int4
  if (is_per_tensor && !is_int4_type) {
    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    params_.scaleOffsetEncoding.scale = scales[0];

    if (ort_quant_params->zero_point != nullptr) {
      ORT_RETURN_IF_NOT(zero_points.size() == 1, "Expected one zero-point value");
      params_.scaleOffsetEncoding.offset = zero_points[0];
    } else {
      params_.scaleOffsetEncoding.offset = 0;
    }
  } else if (is_per_tensor && is_int4_type) {
    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
    params_.bwScaleOffsetEncoding.bitwidth = 4;
    params_.bwScaleOffsetEncoding.scale = scales[0];

    if (ort_quant_params->zero_point != nullptr) {
      ORT_RETURN_IF_NOT(zero_points.size() == 1, "Expected one zero-point value");
      params_.bwScaleOffsetEncoding.offset = zero_points[0];
    } else {
      params_.bwScaleOffsetEncoding.offset = 0;
    }
  } else if (!is_per_tensor && is_int4_type) {
    const std::vector<int64_t> io_shape = io_def.shape;
    ORT_RETURN_IF(io_shape.empty(), "Input/output tensor proto must have a shape");
    const int32_t io_rank = static_cast<int32_t>(io_shape.size());

    constexpr int64_t DEFAULT_QDQ_AXIS = 1;
    int64_t axis = ort_quant_params->axis.value_or(DEFAULT_QDQ_AXIS);
    if (axis < 0) {
      axis += io_rank;
    }
    ORT_RETURN_IF_NOT(axis >= 0 && axis < io_rank,
                      "Quantization axis must be within the range [0, rank - 1]");

    const size_t num_elems = scales.size();
    const bool no_zero_points = zero_points.empty();
    ORT_RETURN_IF_NOT(num_elems > 1, "Expected more than one scale value");
    ORT_RETURN_IF_NOT(no_zero_points || zero_points.size() == num_elems,
                      "Expected the same number of zero-points and scales for per-channel quantization");

    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
    params_.bwAxisScaleOffsetEncoding.axis = static_cast<int32_t>(axis);
    params_.bwAxisScaleOffsetEncoding.bitwidth = 4;
    params_.bwAxisScaleOffsetEncoding.numElements = static_cast<uint32_t>(num_elems);

    const size_t num_scale_bytes = num_elems * sizeof(float);
    const size_t num_zp_bytes = num_elems * sizeof(int32_t);
    const size_t num_bytes = num_scale_bytes + num_zp_bytes;
    constexpr std::uintptr_t align = alignof(float);
    per_channel_data_ = std::make_unique<char[]>(num_bytes + align);

    char* scales_begin = ALIGN_PTR_UP(per_channel_data_.get(), align, char*);
    char* zps_begin = scales_begin + num_scale_bytes;
    gsl::span<float> scales_span(reinterpret_cast<float*>(scales_begin), num_elems);
    gsl::span<int32_t> zps_span(reinterpret_cast<int32_t*>(zps_begin), num_elems);

    for (size_t i = 0; i < num_elems; i++) {
      scales_span[i] = scales[i];
      zps_span[i] = no_zero_points ? 0 : zero_points[i];
    }

    params_.bwAxisScaleOffsetEncoding.scales = scales_span.data();
    params_.bwAxisScaleOffsetEncoding.offsets = zps_span.data();
  } else if (!is_per_tensor && !is_int4_type) {
    const std::vector<int64_t> io_shape = io_def.shape;
    ORT_RETURN_IF(io_shape.empty(), "Input/output tensor proto must have a shape");
    const int32_t io_rank = static_cast<int32_t>(io_shape.size());

    constexpr int64_t DEFAULT_QDQ_AXIS = 1;
    int64_t axis = ort_quant_params->axis.value_or(DEFAULT_QDQ_AXIS);
    if (axis < 0) {
      axis += io_rank;
    }
    ORT_RETURN_IF_NOT(axis >= 0 && axis < io_rank,
                      "Quantization axis must be within the range [0, rank - 1]");

    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;

    const size_t num_elems = scales.size();
    const bool no_zero_points = zero_points.empty();
    ORT_RETURN_IF_NOT(num_elems > 1, "Expected more than one scale value");
    ORT_RETURN_IF_NOT(no_zero_points || zero_points.size() == num_elems,
                      "Expected the same number of zero-points and scales for per-channel quantization");

    const size_t num_bytes = num_elems * sizeof(Qnn_ScaleOffset_t);
    constexpr std::uintptr_t align = alignof(Qnn_ScaleOffset_t);
    per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
    Qnn_ScaleOffset_t* aligned_dst = ALIGN_PTR_UP(per_channel_data_.get(), align, Qnn_ScaleOffset_t*);
    gsl::span<Qnn_ScaleOffset_t> data_span(aligned_dst, num_elems);

    for (size_t i = 0; i < num_elems; i++) {
      data_span[i].scale = scales[i];
      data_span[i].offset = no_zero_points ? 0 : zero_points[i];
    }

    params_.axisScaleOffsetEncoding.axis = static_cast<int32_t>(axis);
    params_.axisScaleOffsetEncoding.numScaleOffsets = static_cast<uint32_t>(num_elems);
    params_.axisScaleOffsetEncoding.scaleOffset = data_span.data();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected tensor kind for QuantParamsWrapper::Init()");
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
