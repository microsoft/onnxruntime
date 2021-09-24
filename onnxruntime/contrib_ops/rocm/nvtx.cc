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
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create()),
    NvtxPush);

ONNX_OPERATOR_KERNEL_EX(
    NvtxPop,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create()),
    NvtxPop);

Status NvtxPush::ComputeInternal(OpKernelContext* context) const {
  std::cout << "executing nvtx push" << std::endl;
  roctxRangePushA("myrange");

  return Status::OK();
}

Status NvtxPop::ComputeInternal(OpKernelContext* context) const {
  std::cout << "executing nvtx pop" << std::endl;
  roctxRangePop();

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
