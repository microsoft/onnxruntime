// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ENABLE_TRAINING_TORCH_INTEROP

#include "core/language_interop_ops/torch/cpu/torch_custom_function_kernel.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "core/language_interop_ops/torch/refcount_tracker.h"

using namespace onnxruntime::language_interop_ops::torch;

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    PythonOp,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOp);

ONNX_OPERATOR_KERNEL_EX(
    PythonOpGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOpGrad);

Status PythonOp::Compute(OpKernelContext* context) const {
  void* diff_ctx = nullptr;
  std::vector<OrtValue> returned_ortvalues;
  RunForward(context, &diff_ctx, returned_ortvalues);

  SetOutputs(context, diff_ctx, returned_ortvalues);

  RefCountTracker::GetInstance().DumpDetails("Forward Kernel Completed");
  return Status::OK();
}

Status PythonOpGrad::Compute(OpKernelContext* context) const {
  std::vector<OrtValue> returned_ortvalues;
  RunBackward(context, returned_ortvalues);

  SetOutputs(context, returned_ortvalues);

  RefCountTracker::GetInstance().DumpDetails("Backward Kernel Completed");
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif
