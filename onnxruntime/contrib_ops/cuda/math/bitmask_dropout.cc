// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/dropout.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
// Bitmask tensor is uint_32 type.
using BitmaskElementType = uint32_t;
}  // namespace

ONNX_OPERATOR_KERNEL_EX(BitmaskDropout, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("T3", DataTypeImpl::GetTensorType<BitmaskElementType>())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .InputMemoryType(OrtMemTypeCPUInput, 2),
                        onnxruntime::cuda::Dropout<true>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
