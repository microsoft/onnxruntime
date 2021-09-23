// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/nvtx.h"
#include "core/providers/common.h"

#include "roctracer/roctx.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {


ONNX_OPERATOR_KERNEL_EX(
    NvtxPush,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()),
    NvtxPush);

ONNX_OPERATOR_KERNEL_EX(
    NvtxPop,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()),
    NvtxPop);

Status NvtxPush::ComputeInternal(OpKernelContext* context) const {
  roctxRangePushA("myrange");
}

Status NvtxPop::ComputeInternal(OpKernelContext* context) const {
  roctxRangePop();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
