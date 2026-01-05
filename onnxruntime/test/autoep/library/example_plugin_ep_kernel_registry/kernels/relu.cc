// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "relu.h"

#include <gsl/span>
#include <algorithm>
#include <cassert>

#include "utils.h"

// Defines a kernel creation function for version 14 of Relu.
ONNX_OPERATOR_KERNEL_EX(
    Relu,
    kOnnxDomain,
    /*version*/ 14,  // Equivalent to start_version: 14, end_version: 14 (inclusive)
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorTypes({ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}))
         .AddInputOutputMutableAlias(0, 0)),
    Relu)

Relu::Relu(const OrtKernelInfo* info, void* /*state*/, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Relu::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Relu>& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Ort::ConstKernelInfo kernel_info(info);
  kernel = std::make_unique<Relu>(info, state, PrivateTag{});
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

template <typename T>
static OrtStatus* ApplyRelu(Ort::KernelContext kernel_context) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  gsl::span<const T> input0;
  std::vector<int64_t> shape0;
  RETURN_IF_ERROR(GetKernelInputDataAndShape<T>(kernel_context, 0, input0, shape0));

  Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
  T* output_data = output.GetTensorMutableData<T>();

  for (size_t i = 0; i < input0.size(); ++i) {
    output_data[i] = std::max(static_cast<T>(0), input0[i]);
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL Relu::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Relu* relu_kernel = static_cast<Relu*>(this_ptr);
  Ort::KernelContext kernel_context(kernel_ctx);
  static_cast<void>(relu_kernel->info_);  // NOTE: Unused in this example.

  Ort::ConstValue input = kernel_context.GetInput(0);
  auto type_shape = input.GetTensorTypeAndShapeInfo();
  auto elem_type = type_shape.GetElementType();

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    ApplyRelu<float>(kernel_context);
  } else {
    assert(elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ApplyRelu<int64_t>(kernel_context);
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL Relu::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Relu*>(this_ptr);
}
