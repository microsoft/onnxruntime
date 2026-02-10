// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor_converter.h"
#include <cstring>
#include <memory>

namespace onnxruntime {
namespace telum {

Status TensorConverter::ConvertToZTensor(const Tensor& ort_tensor,
                                         zdnn_ztensor& ztensor,
                                         zdnn_data_layouts layout) {
  return ConvertToZTensorWithShape(ort_tensor, ort_tensor.Shape(), ztensor, layout);
}

Status TensorConverter::ConvertToZTensorWithShape(const Tensor& ort_tensor,
                                                  const TensorShape& logical_shape,
                                                  zdnn_ztensor& ztensor,
                                                  zdnn_data_layouts layout) {
  if (ort_tensor.Shape().Size() != logical_shape.Size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Logical shape element count (", logical_shape.Size(),
                           ") does not match ORT tensor element count (", ort_tensor.Shape().Size(), ")");
  }

  return ConvertRawToZTensor(ort_tensor.DataRaw(),
                             ort_tensor.GetElementType(),
                             logical_shape,
                             ztensor,
                             layout);
}

Status TensorConverter::ConvertRawToZTensor(const void* raw_data,
                                            int32_t ort_type,
                                            const TensorShape& logical_shape,
                                            zdnn_ztensor& ztensor,
                                            zdnn_data_layouts layout) {
  ORT_RETURN_IF_NOT(raw_data != nullptr, "Raw data pointer is null");

  // Validate static shape
  ORT_RETURN_IF_ERROR(ValidateStaticShape(logical_shape));

  // Validate shape is compatible with layout
  ORT_RETURN_IF_ERROR(ValidateShapeForLayout(logical_shape, layout));

  // Allocate descriptors on the heap so they outlive this function. zDNN stores pointers to them in the ztensor.
  auto pre_desc = std::make_unique<zdnn_tensor_desc>();
  auto tfrmd_desc = std::make_unique<zdnn_tensor_desc>();

  ORT_RETURN_IF_ERROR(InitializeDescriptors(logical_shape, ort_type, layout, *pre_desc, *tfrmd_desc));

  // Initialize ztensor structure
  zdnn_init_ztensor(pre_desc.get(), tfrmd_desc.get(), &ztensor);

  // Allocate buffer for ztensor
  zdnn_status status = zdnn_allochelper_ztensor(&ztensor);
  if (status != ZDNN_OK) {
    // Ensure we don't return a ztensor with dangling descriptor pointers (owned by the local unique_ptrs).
    if (ztensor.buffer != nullptr) {
      zdnn_free_ztensor_buffer(&ztensor);
      ztensor.buffer = nullptr;
    }
    ztensor.pre_transformed_desc = nullptr;
    ztensor.transformed_desc = nullptr;
    return CheckZDNNStatus(status, "zdnn_allochelper_ztensor");
  }

  // Transform data from ORT format to zDNN format
  Status st = TransformData(raw_data, ort_type, ztensor);
  if (!st.IsOK()) {
    // If transform fails, free buffer before returning.
    if (ztensor.buffer != nullptr) {
      zdnn_free_ztensor_buffer(&ztensor);
      ztensor.buffer = nullptr;
    }
    // As above, avoid leaving dangling pointers in ztensor.
    ztensor.pre_transformed_desc = nullptr;
    ztensor.transformed_desc = nullptr;
    return st;
  }

  // Success: transfer descriptor ownership to the ztensor.
  (void)pre_desc.release();
  (void)tfrmd_desc.release();

  return Status::OK();
}

Status TensorConverter::ConvertFromZTensor(const zdnn_ztensor& ztensor,
                                           Tensor& ort_tensor) {
  // Validate ztensor is transformed
  if (!ztensor.is_transformed) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "Cannot convert untransformed ztensor to ORT tensor");
  }

  // Transform data back to original format
  zdnn_status status = zdnn_transform_origtensor(&ztensor, ort_tensor.MutableDataRaw());
  ORT_RETURN_IF_ERROR(CheckZDNNStatus(status, "zdnn_transform_origtensor"));

  return Status::OK();
}

Status TensorConverter::InitZTensorForOutput(const Tensor& ort_tensor,
                                             zdnn_ztensor& ztensor,
                                             zdnn_data_layouts layout) {
  return InitZTensorForOutputWithShape(ort_tensor, ort_tensor.Shape(), ztensor, layout);
}

Status TensorConverter::InitZTensorForOutputWithShape(const Tensor& ort_tensor,
                                                      const TensorShape& logical_shape,
                                                      zdnn_ztensor& ztensor,
                                                      zdnn_data_layouts layout) {
  if (ort_tensor.Shape().Size() != logical_shape.Size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Logical shape element count (", logical_shape.Size(),
                           ") does not match ORT tensor element count (", ort_tensor.Shape().Size(), ")");
  }

  // Validate static shape
  ORT_RETURN_IF_ERROR(ValidateStaticShape(logical_shape));

  // Validate shape is compatible with layout
  ORT_RETURN_IF_ERROR(ValidateShapeForLayout(logical_shape, layout));

  // Allocate descriptors on the heap so they outlive this function. zDNN stores pointers to them in the ztensor.
  auto pre_desc = std::make_unique<zdnn_tensor_desc>();
  auto tfrmd_desc = std::make_unique<zdnn_tensor_desc>();

  ORT_RETURN_IF_ERROR(InitializeDescriptors(logical_shape, ort_tensor.GetElementType(), layout, *pre_desc, *tfrmd_desc));

  // Initialize ztensor structure
  zdnn_init_ztensor(pre_desc.get(), tfrmd_desc.get(), &ztensor);

  // Allocate buffer for ztensor
  zdnn_status status = zdnn_allochelper_ztensor(&ztensor);
  if (status != ZDNN_OK) {
    if (ztensor.buffer != nullptr) {
      zdnn_free_ztensor_buffer(&ztensor);
      ztensor.buffer = nullptr;
    }
    ztensor.pre_transformed_desc = nullptr;
    ztensor.transformed_desc = nullptr;
    return CheckZDNNStatus(status, "zdnn_allochelper_ztensor");
  }

  // Success: transfer descriptor ownership to the ztensor.
  (void)pre_desc.release();
  (void)tfrmd_desc.release();

  return Status::OK();
}

Status TensorConverter::InitZTensorForOutputWithShapeAndType(int32_t ort_type,
                                                             const TensorShape& logical_shape,
                                                             zdnn_ztensor& ztensor,
                                                             zdnn_data_layouts layout) {
  // Validate static shape
  ORT_RETURN_IF_ERROR(ValidateStaticShape(logical_shape));

  // Validate shape is compatible with layout
  ORT_RETURN_IF_ERROR(ValidateShapeForLayout(logical_shape, layout));

  // Allocate descriptors on the heap so they outlive this function. zDNN stores pointers to them in the ztensor.
  auto pre_desc = std::make_unique<zdnn_tensor_desc>();
  auto tfrmd_desc = std::make_unique<zdnn_tensor_desc>();

  ORT_RETURN_IF_ERROR(InitializeDescriptors(logical_shape, ort_type, layout, *pre_desc, *tfrmd_desc));

  // Initialize ztensor structure
  zdnn_init_ztensor(pre_desc.get(), tfrmd_desc.get(), &ztensor);

  // Allocate buffer for ztensor
  zdnn_status status = zdnn_allochelper_ztensor(&ztensor);
  if (status != ZDNN_OK) {
    if (ztensor.buffer != nullptr) {
      zdnn_free_ztensor_buffer(&ztensor);
      ztensor.buffer = nullptr;
    }
    ztensor.pre_transformed_desc = nullptr;
    ztensor.transformed_desc = nullptr;
    return CheckZDNNStatus(status, "zdnn_allochelper_ztensor");
  }

  // Success: transfer descriptor ownership to the ztensor.
  (void)pre_desc.release();
  (void)tfrmd_desc.release();

  return Status::OK();
}

zdnn_data_layouts TensorConverter::GetLayoutForShape(const TensorShape& shape) {
  size_t rank = shape.NumDimensions();

  switch (rank) {
    case 1:
      return ZDNN_1D;
    case 2:
      return ZDNN_2D;
    case 3:
      return ZDNN_3D;
    case 4:
      return ZDNN_4D;
    default:
      return ZDNN_2D;  // Default fallback
  }
}

Status TensorConverter::ValidateShapeForLayout(const TensorShape& shape,
                                               zdnn_data_layouts layout) {
  size_t rank = shape.NumDimensions();

  // Validate rank matches layout expectations
  switch (layout) {
    case ZDNN_1D:
      if (rank != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "ZDNN_1D layout requires 1D tensor, got rank ", rank);
      }
      break;
    case ZDNN_2D:
    case ZDNN_2DS:
      if (rank != 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "ZDNN_2D layout requires 2D tensor, got rank ", rank);
      }
      break;
    case ZDNN_3D:
    case ZDNN_3DS:
      if (rank != 3) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "ZDNN_3D layout requires 3D tensor, got rank ", rank);
      }
      break;
    case ZDNN_4D:
    case ZDNN_4DS:
    case ZDNN_NHWC:
    case ZDNN_NCHW:
      if (rank != 4) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "ZDNN_4D layout requires 4D tensor, got rank ", rank);
      }
      break;
    default:
      break;
  }

  // Validate dimensions don't exceed hardware limits
  uint32_t max_dim = zdnn_get_nnpa_max_dim_idx_size();
  for (auto dim : shape.GetDims()) {
    if (static_cast<uint32_t>(dim) > max_dim) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Dimension ", dim, " exceeds zDNN maximum ", max_dim);
    }
  }

  return Status::OK();
}

Status TensorConverter::InitializeDescriptors(const Tensor& ort_tensor,
                                              zdnn_data_layouts layout,
                                              zdnn_tensor_desc& pre_desc,
                                              zdnn_tensor_desc& tfrmd_desc) {
  return InitializeDescriptors(ort_tensor.Shape(), ort_tensor.GetElementType(), layout, pre_desc, tfrmd_desc);
}

Status TensorConverter::InitializeDescriptors(const TensorShape& logical_shape,
                                              int32_t ort_type,
                                              zdnn_data_layouts layout,
                                              zdnn_tensor_desc& pre_desc,
                                              zdnn_tensor_desc& tfrmd_desc) {
  const auto& dims = logical_shape.GetDims();
  zdnn_data_types zdnn_type = MapONNXTypeToZDNN(ort_type);

  // Initialize pre-transformed descriptor based on rank
  switch (dims.size()) {
    case 1:
      zdnn_init_pre_transformed_desc(layout, zdnn_type, &pre_desc,
                                     static_cast<uint32_t>(dims[0]));
      break;
    case 2:
      zdnn_init_pre_transformed_desc(layout, zdnn_type, &pre_desc,
                                     static_cast<uint32_t>(dims[0]),
                                     static_cast<uint32_t>(dims[1]));
      break;
    case 3:
      zdnn_init_pre_transformed_desc(layout, zdnn_type, &pre_desc,
                                     static_cast<uint32_t>(dims[0]),
                                     static_cast<uint32_t>(dims[1]),
                                     static_cast<uint32_t>(dims[2]));
      break;
    case 4:
      zdnn_init_pre_transformed_desc(layout, zdnn_type, &pre_desc,
                                     static_cast<uint32_t>(dims[0]),
                                     static_cast<uint32_t>(dims[1]),
                                     static_cast<uint32_t>(dims[2]),
                                     static_cast<uint32_t>(dims[3]));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported tensor rank: ", dims.size());
  }

  // Generate transformed descriptor
  zdnn_status status = zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc);
  ORT_RETURN_IF_ERROR(CheckZDNNStatus(status, "zdnn_generate_transformed_desc"));

  return Status::OK();
}

Status TensorConverter::TransformData(const void* ort_data,
                                      int32_t ort_type,
                                      zdnn_ztensor& ztensor) {
  // Transform data using zDNN API
  zdnn_status status = zdnn_transform_ztensor(&ztensor, ort_data);
  ORT_RETURN_IF_ERROR(CheckZDNNStatus(status, "zdnn_transform_ztensor"));

  return Status::OK();
}

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
