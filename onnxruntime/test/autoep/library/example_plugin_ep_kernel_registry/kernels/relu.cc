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
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddInputOutputMutableAlias(0, 0)),
    Relu)

Relu::Relu(const OrtKernelInfo* info, void* /*state*/, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl members to NULL/zero
      info_{info} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Relu::CreateKernelImpl(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Ort::ConstKernelInfo kernel_info(info);
  auto relu_kernel = std::make_unique<Relu>(info, state, PrivateTag{});

  kernel = relu_kernel.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL Relu::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Relu* relu_kernel = static_cast<Relu*>(this_ptr);
  Ort::KernelContext kernel_context(kernel_ctx);
  static_cast<void>(relu_kernel->info_);  // NOTE: Unused in this example.

  gsl::span<const float> input0;
  std::vector<int64_t> shape0;
  RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));

  Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
  float* output_data = output.GetTensorMutableData<float>();

  for (size_t i = 0; i < input0.size(); ++i) {
    output_data[i] = std::max(0.0f, input0[i]);
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL Relu::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Relu*>(this_ptr);
}
