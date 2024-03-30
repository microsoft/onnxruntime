// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "QnnTypes.h"
#include "core/common/common.h"
#include "core/common/gsl.h"
#include "core/framework/node_unit.h"

#include <memory>

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;  // Forward-declare

class QnnQuantParamsWrapper {
 public:
  QnnQuantParamsWrapper() : params_(QNN_QUANTIZE_PARAMS_INIT) {}
  QnnQuantParamsWrapper(const Qnn_QuantizeParams_t& params);

  QnnQuantParamsWrapper(const QnnQuantParamsWrapper& other);
  QnnQuantParamsWrapper& operator=(const QnnQuantParamsWrapper& other);

  QnnQuantParamsWrapper(QnnQuantParamsWrapper&& other) = default;
  QnnQuantParamsWrapper& operator=(QnnQuantParamsWrapper&& other) = default;

  QnnQuantParamsWrapper(float scale, int32_t offset = 0);

  Qnn_QuantizeParams_t& Get() { return params_; }
  const Qnn_QuantizeParams_t& GetConst() const { return params_; }

  Status Init(const Qnn_QuantizeParams_t& params);
  Status Init(const QnnModelWrapper& qnn_model_wrapper, const NodeUnitIODef& io_def);

  QnnQuantParamsWrapper Copy() const;

  bool IsNotQuantized() const {
    return params_.encodingDefinition != QNN_DEFINITION_DEFINED;
  }

  bool IsPerTensorQuantization(bool include_bw = false) const {
    return params_.encodingDefinition == QNN_DEFINITION_DEFINED &&
           (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
            (include_bw && params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET));
  }

  bool IsPerAxisQuantization(bool include_bw = false) const {
    return params_.encodingDefinition == QNN_DEFINITION_DEFINED &&
           (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET ||
            (include_bw && params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET));
  }

  template <typename IntType>
  Status HandleTranspose(gsl::span<const IntType> perm) {
    if (!IsPerAxisQuantization(true)) {
      return Status::OK();
    }

    if (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      ORT_RETURN_IF_NOT(static_cast<size_t>(params_.axisScaleOffsetEncoding.axis) < perm.size(),
                        "Axis value is out of range of the provided permutation");
      const int32_t new_axis = static_cast<int32_t>(perm[params_.axisScaleOffsetEncoding.axis]);
      params_.axisScaleOffsetEncoding.axis = new_axis;
    } else if (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
      ORT_RETURN_IF_NOT(static_cast<size_t>(params_.bwAxisScaleOffsetEncoding.axis) < perm.size(),
                        "Axis value is out of range of the provided permutation");
      const int32_t new_axis = static_cast<int32_t>(perm[params_.bwAxisScaleOffsetEncoding.axis]);
      params_.bwAxisScaleOffsetEncoding.axis = new_axis;
    }

    return Status::OK();
  }

  template <typename IntType>
  Status HandleUnsqueeze(gsl::span<const IntType> orig_shape,
                         gsl::span<const IntType> new_shape) {
    if (!IsPerAxisQuantization(true)) {
      return Status::OK();
    }

    ORT_RETURN_IF_NOT(orig_shape.size() < new_shape.size(), "Expected unsqueezed shape to have a greater rank.");

    // Get the axis value.
    int32_t axis = 0;
    if (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      axis = params_.axisScaleOffsetEncoding.axis;
    } else if (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
      axis = params_.bwAxisScaleOffsetEncoding.axis;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Unhandled quantization encoding: ", params_.quantizationEncoding);
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

    // Set new axis.
    if (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      params_.axisScaleOffsetEncoding.axis = static_cast<int32_t>(j);
    } else if (params_.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
      params_.bwAxisScaleOffsetEncoding.axis = static_cast<int32_t>(j);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Unhandled quantization encoding: ", params_.quantizationEncoding);
    }

    return Status::OK();
  }

 private:
  Qnn_QuantizeParams_t params_;
  std::unique_ptr<char[]> scale_offset_data_;  // Stores per-axis scales and offsets
};

}  // namespace qnn
}  // namespace onnxruntime
