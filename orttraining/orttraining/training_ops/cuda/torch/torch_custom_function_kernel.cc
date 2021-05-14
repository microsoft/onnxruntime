// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/torch/torch_custom_function_kernel.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/language_interop_ops/torch/custom_function_register.h"

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
  std::vector<OrtValue*> args = contrib::CreateOrtValueArgs(context, 0);
  // Place holder for Python returned values.
  std::vector<void*> returned_args;

  // Invoke python calls.
  std::string err;
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().Forward(
      onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool ::GetInstance()
          .GetForwardCore(name_),
      input_tensor_requires_grads_,
      args,
      arg_positions_,
      const_args_,
      const_arg_positions_,
      returned_args,
      is_training_mode_);

  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  // First output of this op is Pytorch autograd's context.
  SetContextOutput(context, returned_args);
  // Other outputs are wrappers of Pytorch tensors.
  SetOtherOutputs(context, returned_args);
  return Status::OK();
}

Status PythonOpGrad::ComputeInternal(OpKernelContext* context) const {
  // Todo(pengwa): perf impact and how much, leave it now to guarantee correctness.
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  auto args = contrib::CreateOrtValueArgs(context, 1);
  // This is called "const" because that's how Pytorch calls all non-tensor inputs.
  auto const_args = CreateConstArgs(context);
  std::vector<void*> returned_args;

  std::string err;
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().Backward(
      onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance()
          .GetBackwardCore(name_),
      input_tensor_requires_grads_,
      args,
      arg_positions_,
      const_args,
      const_arg_positions_,
      returned_args);
  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  SetOutputs(context, returned_args);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
