// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(ATenOp, kMSDomain, 1, kCudaExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        onnxruntime::contrib::ATenOpBase<false>);

ONNX_OPERATOR_KERNEL_EX(ATenOpGrad, kMSDomain, 1, kCudaExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        onnxruntime::contrib::ATenOpBase<true>);

}  // namespace cuda
}  // namespace onnxruntime
