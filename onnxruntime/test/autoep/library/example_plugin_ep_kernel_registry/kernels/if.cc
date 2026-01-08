// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "if.h"

#include "utils.h"

// Defines a kernel creation function for If opset 21
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    If,
    kOnnxDomain,
    /*start version*/ 21, /*end version*/ 22,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If)

// Defines a kernel creation function for If opset 23
ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    /*version*/ 23,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If)

// Defines a kernel creation function for If opset 24
ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    /*version*/ 24,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If)

/*static*/
OrtStatus* If::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<If>& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  OrtKernelImpl* control_flow_kernel = nullptr;

  RETURN_IF_ERROR(Ort::GetEpApi().CreateIfKernel(info, &control_flow_kernel));
  kernel = std::make_unique<If>(info, state, control_flow_kernel, PrivateTag{});

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

If::If(const OrtKernelInfo* info, void* state, OrtKernelImpl* control_flow_kernel, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)},
      control_flow_kernel_{control_flow_kernel} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  GetControlFlowKernel = GetControlFlowKernelImpl;
}

If::~If() {
  Ort::GetEpApi().ReleaseKernelImpl(control_flow_kernel_);
}

/*static*/
OrtStatus* ORT_API_CALL If::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  If* if_kernel = static_cast<If*>(this_ptr);
  static_cast<void>(if_kernel->info_);                // NOTE: Unused in this example.
  static_cast<void>(if_kernel->data_transfer_impl_);  // NOTE: Unused in this example.

  return if_kernel->control_flow_kernel_->Compute(if_kernel->control_flow_kernel_, kernel_ctx);
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL If::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<If*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL If::GetControlFlowKernelImpl(OrtKernelImpl* this_ptr, OrtKernelImpl** out) noexcept {
  If* if_kernel = static_cast<If*>(this_ptr);
  *out = if_kernel->control_flow_kernel_;
  return nullptr;
}
