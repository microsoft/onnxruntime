// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "contrib_ops/cpu/aten_ops/aten_op.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ATen, kPytorchAtenDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
    onnxruntime::contrib::ATen);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
