// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITON

#include "core/providers/shared_library/provider_api.h"
#include "orttraining/training_ops/cpu/triton/triton_op.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(TritonOp, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        onnxruntime::contrib::TritonOp);

}  // namespace cuda
}  // namespace onnxruntime

#endif
