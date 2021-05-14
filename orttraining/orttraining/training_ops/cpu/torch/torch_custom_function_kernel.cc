// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/language_interop_ops/torch/custom_function_register.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    PythonOp,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOp);

ONNX_OPERATOR_KERNEL_EX(
    PythonOpGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOpGrad);

Status PythonOp::Compute(OpKernelContext* context) const {
  // Create non-constant arguments for calling Python function.
  // Constant arguments are created in ctor.
  std::vector<OrtValue*> args = CreateOrtValueArgs(context, 0);
  // Placeholder for Python returned values.
  std::vector<void*> returned_args;

  // Invoke Python calls.
  std::string err;
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().Forward(
      onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance()
          .GetForwardCore(name_),
      input_tensor_requires_grads_,
      args,
      arg_positions_,
      const_args_,
      const_arg_positions_,
      returned_args,
      is_training_mode_);

  // First output of this op is Pytorch autograd's context.
  SetContextOutput(context, returned_args);
  // Other outputs are wrappers of Pytorch tensors.
  SetOtherOutputs(context, returned_args);
  return Status::OK();
}

Status PythonOpGrad::Compute(OpKernelContext* context) const {
  auto args = CreateOrtValueArgs(context, 1);
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

  SetOutputs(context, returned_args);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
