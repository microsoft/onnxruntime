// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "../telum_common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Utility class for converting between ORT tensors and zDNN ztensors
 *
 * This class handles the conversion between ONNX Runtime's tensor format
 * and zDNN's ztensor format, including data layout transformations and
 * memory management.
 */
class TensorConverter {
 public:
  /**
   * @brief Convert ORT tensor to zDNN ztensor
   *
   * @param ort_tensor Input ORT tensor
   * @param ztensor Output zDNN ztensor (must be pre-allocated)
   * @param layout Desired zDNN data layout
   * @return Status indicating success or failure
   */
  static Status ConvertToZTensor(const Tensor& ort_tensor,
                                 zdnn_ztensor& ztensor,
                                 zdnn_data_layouts layout);

  /**
   * @brief Convert zDNN ztensor back to ORT tensor
   *
   * @param ztensor Input zDNN ztensor
   * @param ort_tensor Output ORT tensor (must be pre-allocated)
   * @return Status indicating success or failure
   */
  static Status ConvertFromZTensor(const zdnn_ztensor& ztensor,
                                   Tensor& ort_tensor);

  /**
   * @brief Initialize zDNN ztensor for output
   *
   * @param ort_tensor ORT tensor to base initialization on
   * @param ztensor Output zDNN ztensor to initialize
   * @param layout Desired zDNN data layout
   * @return Status indicating success or failure
   */
  static Status InitZTensorForOutput(const Tensor& ort_tensor,
                                     zdnn_ztensor& ztensor,
                                     zdnn_data_layouts layout);

  /**
   * @brief Get zDNN layout for given tensor shape
   *
   * @param shape Tensor shape
   * @return Appropriate zDNN data layout
   */
  static zdnn_data_layouts GetLayoutForShape(const TensorShape& shape);

  /**
   * @brief Validate tensor shape is compatible with zDNN
   *
   * @param shape Tensor shape to validate
   * @param layout Expected zDNN layout
   * @return Status indicating if shape is valid
   */
  static Status ValidateShapeForLayout(const TensorShape& shape,
                                       zdnn_data_layouts layout);

 private:
  /**
   * @brief Initialize zDNN tensor descriptor from ORT tensor
   *
   * @param ort_tensor Input ORT tensor
   * @param layout Desired zDNN layout
   * @param pre_desc Output pre-transformed descriptor
   * @param tfrmd_desc Output transformed descriptor
   * @return Status indicating success or failure
   */
  static Status InitializeDescriptors(const Tensor& ort_tensor,
                                      zdnn_data_layouts layout,
                                      zdnn_tensor_desc& pre_desc,
                                      zdnn_tensor_desc& tfrmd_desc);

  /**
   * @brief Transform data from ORT format to zDNN format
   *
   * @param ort_data Source data pointer
   * @param ort_type ORT data type
   * @param ztensor Destination ztensor
   * @return Status indicating success or failure
   */
  static Status TransformData(const void* ort_data,
                              int32_t ort_type,
                              zdnn_ztensor& ztensor);
};

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
