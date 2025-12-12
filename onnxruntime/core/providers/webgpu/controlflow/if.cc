// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/controlflow/if.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  1, 10,
                                  kWebGpuExecutionProvider,
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
                                  kWebGpuExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  If);

// opset-13 supports sequence type for If's subgraph outputs
ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  13, 18,
                                  kWebGpuExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      // Support sequence/optional tensors when all WebGPU infra
                                      // (including tests runner) supports it
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  If);

// opset-19 supports float8
ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  19, 20,
                                  kWebGpuExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      // Support sequence/optional tensors when all WebGPU infra
                                      // (including tests runner) supports it
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  If);

ONNX_OPERATOR_KERNEL_EX(If,
                        kOnnxDomain,
                        21,
                        kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            // Support sequence/optional tensors when all WebGPU infra
                            // (including tests runner) supports it
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                        If);

Status If::Compute(OpKernelContext* ctx) const {
#if defined(BUILD_WEBGPU_EP_STATIC_LIB)
  // call the base CPU version.
  return onnxruntime::If::Compute(ctx);
#else
  // TODO(fs-eire): implement WebGPU If kernel
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "If operator is not implemented for WebGPU EP yet.");
#endif
}

}  // namespace webgpu
}  // namespace onnxruntime
