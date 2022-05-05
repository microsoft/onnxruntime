// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "internal_testing_ep_static_kernels.h"

#include "core/framework/utils.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/utils.h"

namespace onnxruntime {
namespace internal_testing_ep {

// can't use 'utils::kInternalTestingExecutionProvider' in the macro so redefine here to a name without '::'
constexpr const char* internal_testing_ep = utils::kInternalTestingExecutionProvider;

// For each layout sensitive kernel, register the NHWC implementation (kMSInternalNHWCDomain)
// and a stub NCHW (kOnnxDomain) version.
//
// The first call to GetCapability will match the NCHW version.
// The layout transform will create a replacement node in the NHWC domain.
// The kernel matching in SessionState finalization will use the NHWC kernel to run the node.

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Conv, kMSInternalNHWCDomain, 1, 10, internal_testing_ep,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  Conv);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Conv, kOnnxDomain, 1, 10, internal_testing_ep,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  utils::InvalidNchwKernel);

ONNX_OPERATOR_KERNEL_EX(Conv, kMSInternalNHWCDomain, 11, internal_testing_ep,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Conv);
ONNX_OPERATOR_KERNEL_EX(Conv, kOnnxDomain, 11, internal_testing_ep,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        utils::InvalidNchwKernel);

//
// Kernel implementation example

Status Conv::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // this is in NHWC format
  std::cout << "Compute called with input shape of " << X.Shape() << "\n";
  ORT_NOT_IMPLEMENTED("TODO: add NHWC implementation here.");
}

}  // namespace internal_testing_ep
}  // namespace onnxruntime
