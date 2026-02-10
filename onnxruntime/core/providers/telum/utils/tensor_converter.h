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
   * @brief Convert ORT tensor to zDNN ztensor, using an alternate logical shape.
   *
   * This is useful when the ORT tensor shape can be safely re-interpreted as a different
   * shape for a given zDNN op (e.g., collapsing multiple batch dims into one stack dim).
   *
   * @param ort_tensor Input ORT tensor
   * @param logical_shape Shape to use for zDNN descriptors (must have same element count as ort_tensor)
   * @param ztensor Output zDNN ztensor (must be pre-allocated)
   * @param layout Desired zDNN data layout (must be compatible with logical_shape rank)
   * @return Status indicating success or failure
   */
  static Status ConvertToZTensorWithShape(const Tensor& ort_tensor,
                                         const TensorShape& logical_shape,
                                         zdnn_ztensor& ztensor,
                                         zdnn_data_layouts layout);

  /**
   * @brief Convert raw host data (in ORT/ONNX conventional layout) to a transformed zDNN ztensor.
   *
   * @param raw_data Pointer to raw input data
   * @param ort_type ONNX TensorProto data type value (e.g., FLOAT, FLOAT16, BFLOAT16)
   * @param logical_shape Shape to use for zDNN descriptors
   * @param ztensor Output zDNN ztensor (must be pre-allocated)
   * @param layout Desired zDNN data layout
   * @return Status indicating success or failure
   */
  static Status ConvertRawToZTensor(const void* raw_data,
                                   int32_t ort_type,
                                   const TensorShape& logical_shape,
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
   * @brief Initialize zDNN ztensor for an ORT output tensor, using an alternate logical shape.
   *
   * @param ort_tensor ORT output tensor (used for element type only)
   * @param logical_shape Shape to use for zDNN descriptors (must have same element count as ort_tensor)
   * @param ztensor Output zDNN ztensor to initialize
   * @param layout Desired zDNN data layout
   * @return Status indicating success or failure
   */
  static Status InitZTensorForOutputWithShape(const Tensor& ort_tensor,
                                             const TensorShape& logical_shape,
                                             zdnn_ztensor& ztensor,
                                             zdnn_data_layouts layout);

  /**
   * @brief Initialize a zDNN ztensor for an intermediate/output tensor given an ONNX element type.
   *
   * This is useful for zDNN ops that produce intermediate outputs that are not directly backed by an ORT Tensor
   * (e.g., mean/variance outputs for normalization).
   *
   * @param ort_type ONNX TensorProto data type value (e.g., FLOAT, FLOAT16, BFLOAT16)
   * @param logical_shape Shape to use for zDNN descriptors
   * @param ztensor Output zDNN ztensor to initialize
   * @param layout Desired zDNN data layout
   * @return Status indicating success or failure
   */
  static Status InitZTensorForOutputWithShapeAndType(int32_t ort_type,
                                                    const TensorShape& logical_shape,
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

  static Status InitializeDescriptors(const TensorShape& logical_shape,
                                      int32_t ort_type,
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
