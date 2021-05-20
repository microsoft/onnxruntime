// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/language_interop_ops/torch/custom_function_register.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/language_interop_ops/torch/refcount_tracker.h"
#include "orttraining/training_ops/cuda/torch/torch_custom_function_kernel.h"

using namespace onnxruntime::language_interop_ops::torch;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    PythonOp,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOp);

ONNX_OPERATOR_KERNEL_EX(
    PythonOpGrad,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOpGrad);

Status PythonOp::ComputeInternal(OpKernelContext* context) const {
  // Todo(pengwa): perf impact and how much, leave it now to guarantee correctness.
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  // Create non-constant arguments for calling Python function.
  // Constant arguments are created in ctor.
  std::vector<OrtValue*> args = contrib::CreateOrtValueArgs(context, 0, context->InputCount());

  void* diff_ctx = nullptr;
  std::vector<OrtValue> returned_ortvalues;

  // Invoke python calls.
  std::string err;
  TorchProxy::GetInstance().Forward(
      OrtTorchFunctionPool::GetInstance().GetForwardCore(name_),
      input_tensor_requires_grads_,
      args,
      arg_positions_,
      const_args_,
      const_arg_positions_,
      &diff_ctx,
      returned_ortvalues,
      is_training_mode_);

  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  ORT_ENFORCE(1 + returned_ortvalues.size() == static_cast<size_t>(context->OutputCount()),
              "Output count mismatch for PythonOp run");
  // First output of this op is Pytorch autograd's context.
  SetContextOutput(context, diff_ctx);
  // Other outputs are wrappers of Pytorch tensors.
  SetOtherOutputs(context, returned_ortvalues);

#ifndef NDEBUG
  RefCountTracker::GetInstance().DumpDetails("Forward Kernel Completed");
#endif
  return Status::OK();
}

Status PythonOpGrad::ComputeInternal(OpKernelContext* context) const {
  // Todo(pengwa): perf impact and how much, leave it now to guarantee correctness.
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  auto args = contrib::CreateOrtValueArgs(context, 1, (context->InputCount() - 1) / 2);
  // This is called "const" because that's how Pytorch calls all non-tensor inputs.
  const Tensor* context_id_tensor = context->Input<Tensor>(0);
  ORT_ENFORCE(context_id_tensor, "Context ID (first input) should not be null.");
  const int64_t* context_index_ptr = context_id_tensor->template Data<int64_t>();
  void* ctx_ptr = OrtTorchFunctionPool::GetInstance().GetContext(*context_index_ptr);
  auto const_args = {ctx_ptr};
  std::vector<OrtValue> returned_ortvalues;

  std::string err;
  TorchProxy::GetInstance().Backward(
      OrtTorchFunctionPool::GetInstance().GetBackwardCore(name_),
      input_tensor_requires_grads_,
      args,
      arg_positions_,
      const_args,
      const_arg_positions_,
      returned_ortvalues);
  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  SetOutputs(context, returned_ortvalues);

#ifndef NDEBUG
  RefCountTracker::GetInstance().DumpDetails("Backward Kernel Completed");
#endif
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
