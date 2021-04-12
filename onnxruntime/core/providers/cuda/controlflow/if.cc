// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/controlflow/if.h"

#include "core/providers/cuda/cuda_fwd.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  1, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  If);

// output shape rules requiring the output shapes of the 'THEN' and 'ELSE'
// branches to be the same were relaxed in opset-11
ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  If);

// sequence tensors were also supported in addition to existing support for tensors in opset-13,
// but we do not support sequence tensors in the cuda If kernel because there are no ops that handle
// sequence tensors on CUDA and supporting it for If doesn't add value while that is the case
ONNX_OPERATOR_KERNEL_EX(If,
                        kOnnxDomain,
                        13,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                        If);

If::If(const OpKernelInfo& info) : OpKernel(info), if_cpu_{CreateOpKernel_CPU_If(info)} {
}

Status If::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this CUDA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = if_cpu_->Compute(ctx);
  return status;
}

}  // namespace cuda
}  // namespace onnxruntime
