// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reshape.h"

#include <gsl/span>
#include <vector>
#include "utils.h"

// ONNX Reshape version 21
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    /*start_version*/ 21, /*end_version (inclusive)*/ 22,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("shape", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddInputOutputAlias(0, 0)
         .SetInputMemType(1, OrtMemTypeCPU)),
    Reshape)

// ONNX Reshape version 23
ONNX_OPERATOR_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    /*version*/ 23,  // Equivalent to start_version: 23, end_version: 23
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("shape", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddInputOutputAlias(0, 0)
         .SetInputMemType(1, OrtMemTypeCPU)),
    Reshape)

// ONNX Reshape version 24
ONNX_OPERATOR_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    /*version*/ 24,  // Equivalent start_version: 24, end_version: 24
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("shape", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddInputOutputAlias(0, 0)
         .SetInputMemType(1, OrtMemTypeCPU)),
    Reshape)

Reshape::Reshape(const OrtKernelInfo* info, void* state, bool allow_zero, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)},
      allow_zero_{allow_zero} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Reshape::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Reshape>& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Ort::ConstKernelInfo kernel_info(info);
  bool allow_zero = kernel_info.GetAttribute<int64_t>("allowzero") == 1;

  kernel = std::make_unique<Reshape>(info, state, allow_zero, PrivateTag{});
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

// Computes the requested shape for the reshape operation.
// Implementation is based on ReshapeHelper in onnxruntime/core/providers/cpu/tensor/reshape_helper.h
static OrtStatus* GetRequestedShape(gsl::span<const int64_t> input_shape, bool allow_zero,
                                    /*out*/ std::vector<int64_t>& requested_shape) {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtApi& ort_api = Ort::GetApi();

  int64_t num_input_elems = 1;
  for (auto dim_val : input_shape) {
    num_input_elems *= dim_val;
  }
  RETURN_IF(num_input_elems == -1, ort_api, "Input tensor must not have dynamic (-1) dimensions.");

  size_t num_dims = requested_shape.size();
  int64_t unknown_dim = -1;
  int64_t size = 1;

  for (size_t i = 0; i < num_dims; i++) {
    RETURN_IF(requested_shape[i] < -1, ort_api, "A dimension cannot be less than -1");

    if (requested_shape[i] == -1) {
      RETURN_IF(unknown_dim != -1, ort_api, "At most one dimension can be -1");
      unknown_dim = static_cast<int64_t>(i);
    } else {
      if (!allow_zero && requested_shape[i] == 0) {
        RETURN_IF(i >= input_shape.size(), ort_api,
                  "The dimension with value zero exceeds the dimension size of the input");
        requested_shape[i] = input_shape[i];
      }

      size *= requested_shape[i];
    }
  }

  if (unknown_dim != -1) {
    // Calculate unknown dimension.
    RETURN_IF(size == 0 || (num_input_elems % size) != 0, ort_api,
              "The input cannot be reshaped to the requested shape");
    requested_shape[unknown_dim] = num_input_elems / size;
  } else {
    // Check if the output shape is valid.
    RETURN_IF(num_input_elems != size, ort_api, "The input cannot be reshaped to the requested shape");
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL Reshape::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Reshape* reshape_kernel = static_cast<Reshape*>(this_ptr);
  static_cast<void>(reshape_kernel->info_);  // NOTE: Unused in this example.

  Ort::KernelContext kernel_context(kernel_ctx);

  // Input[0] has the data to reshape.
  Ort::ConstValue input = kernel_context.GetInput(0);
  auto type_shape_info = input.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_shape = type_shape_info.GetShape();

  // Input[1] has the requested shape for the reshape operation.
  Ort::ConstValue shape_input = kernel_context.GetInput(1);
  gsl::span<const int64_t> shape_input_data;
  std::vector<int64_t> final_shape;

  RETURN_IF_ERROR(GetValueDataAndShape(shape_input, shape_input_data, final_shape));
  RETURN_IF(final_shape.size() != 1, Ort::GetApi(), "A shape tensor must have one dimension");
  RETURN_IF_ERROR(GetRequestedShape(input_shape, reshape_kernel->allow_zero_, final_shape));

  Ort::UnownedValue output = kernel_context.GetOutput(0, final_shape);

  // This kernel aliases the input and output, so a copy is not really necessary.
  // CopyTensor() will not do a copy if the source and destination buffers are the same.
  RETURN_IF_ERROR(CopyTensor(*reshape_kernel->data_transfer_impl_, input, output));
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL Reshape::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Reshape*>(this_ptr);
}
