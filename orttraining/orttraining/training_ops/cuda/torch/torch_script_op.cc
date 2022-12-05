// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "orttraining/training_ops/cpu/torch/torch_script_op.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    TorchScript, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
    onnxruntime::contrib::TorchScript);

}  // namespace cuda
}  // namespace onnxruntime
