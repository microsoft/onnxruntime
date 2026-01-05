// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shape.h"

#include <vector>
#include "utils.h"

// ONNX Shape version 21
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    /*start_version*/ 21, /*end_version (inclusive)*/ 22,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("T1", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .SetOutputMemType(0, OrtMemTypeCPU)),
    Shape)

// ONNX Shape version 23
ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    /*version*/ 23,  // Equivalent to start_version: 23, end_version: 23
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("T1", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .SetOutputMemType(0, OrtMemTypeCPU)),
    Shape)

// ONNX Shape version 24
ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    /*version*/ 24,  // Equivalent start_version: 24, end_version: 24
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("T1", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .SetOutputMemType(0, OrtMemTypeCPU)),
    Shape)

Shape::Shape(const OrtKernelInfo* info, void* state, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Shape::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Shape>& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Ort::ConstKernelInfo kernel_info(info);

  int64_t start = kernel_info.GetAttribute<int64_t>("start");
  int64_t end = 0;
  Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_int64(info, "end", &end)};

  // This example kernel does not support shape slicing.
  RETURN_IF(start != 0 || status.IsOK(), Ort::GetApi(),
            "Example Shape kernel does not support non-default start/end attributes");

  kernel = std::make_unique<Shape>(info, state, PrivateTag{});
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL Shape::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Shape* shape_kernel = static_cast<Shape*>(this_ptr);
  static_cast<void>(shape_kernel->info_);                // NOTE: Unused in this example.
  static_cast<void>(shape_kernel->data_transfer_impl_);  // NOTE: Unused in this example.

  Ort::KernelContext kernel_context(kernel_ctx);

  Ort::ConstValue input = kernel_context.GetInput(0);
  auto type_shape_info = input.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_shape = type_shape_info.GetShape();

  std::vector<int64_t> output_shape = {static_cast<int64_t>(input_shape.size())};
  Ort::UnownedValue output = kernel_context.GetOutput(0, output_shape);
  int64_t* output_data = output.GetTensorMutableData<int64_t>();

  for (size_t i = 0; i < input_shape.size(); i++) {
    output_data[i] = input_shape[i];
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL Shape::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Shape*>(this_ptr);
}
