// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "if.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  1, 10,
                                  kJsExecutionProvider,
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
                                  kJsExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  If);

// opset-13 supports sequence type for If's subgraph outputs
ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  13, 18,
                                  kJsExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      // Support sequence/optional tensors when all JSEP infra
                                      // (including tests runner) supports it
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  If);

// opset-19 supports float8
ONNX_OPERATOR_KERNEL_EX(If,
                        kOnnxDomain,
                        19,
                        kJsExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            // Support sequence/optional tensors when all JSEP infra
                            // (including tests runner) supports it
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                        If);

Status If::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  return onnxruntime::If::Compute(ctx);
}

}  // namespace js
}  // namespace onnxruntime
