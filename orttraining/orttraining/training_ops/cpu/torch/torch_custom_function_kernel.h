// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/language_interop_ops/torch/custom_function_register.h"

namespace onnxruntime {
namespace cuda {

// Pytorch's torch.autograd.Function.apply(...) wrapper.
class PythonOp final : public CudaKernel {
 public:
  PythonOp(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void AddIntScalarArgs();
  void AddInputTupleArgs();
  void AddFloatTupleArgs();
  void AddPointerScalarArgs();
  void CreateConstArgs();
  void CreateArgPositions();

  std::vector<int64_t> const_arg_positions_;
  std::vector<void*> const_args_;
  std::vector<int64_t> arg_positions_;

  // Name of containing class. For example, MyReLU.
  std::string name_;
  int64_t inplace_;
  std::string call_convention_;
  bool is_training_mode_;

  // Attributes of input tensors for calling MyReLU.apply(...).
  // Types. input_tensor_types_[i] is the element type of the i-th tensor.
  std::vector<int64_t> input_tensor_types_;
  // input_tensor_types_[i] indicates if the i-th tensor should have gradient.
  std::vector<int64_t> input_tensor_requires_grads_;

  // Concatenation of all floats from apply(...) 's inputs.
  std::vector<int64_t> input_int_scalars_;
  std::vector<int64_t> input_int_scalar_positions_;

  // Concatenation of all ints from apply(...) 's inputs.
  std::vector<float> input_float_scalars_;
  std::vector<int64_t> input_float_scalar_positions_;

  // Concatenation of all int tuples from apply(...) 's inputs.
  std::vector<int64_t> input_int_tuples_;
  std::vector<int64_t> input_int_tuple_positions_;
  std::vector<int64_t> input_int_tuple_begins_;

  // Concatenation of all float tuples from apply(...) 's inputs.
  std::vector<float> input_float_tuples_;
  std::vector<int64_t> input_float_tuple_positions_;
  std::vector<int64_t> input_float_tuple_begins_;

  std::vector<int64_t> input_pointer_scalars_;
  std::vector<int64_t> input_pointer_scalar_positions_;

  // Output types of MyReLU.apply(...).
  std::vector<int64_t> output_tensor_types_;
  std::vector<int64_t> output_tensor_requires_grads_;
};

// Pytorch's torch.autograd.Function.backward(...) wrapper.
class PythonOpGrad final : public CudaKernel {
 public:
  PythonOpGrad(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void SetPositions();
  // Name of containing class. For example, MyReLU.
  std::string name_;
  // Input types of MyReLU.backward(...).
  std::vector<int64_t> input_tensor_types_;
  // Output types of MyReLU.apply(...).
  std::vector<int64_t> output_tensor_types_;
  std::vector<int64_t> input_tensor_requires_grads_;
  std::vector<int64_t> output_tensor_requires_grads_;
  std::vector<int64_t> arg_positions_;
  std::vector<int64_t> const_arg_positions_;
};

}  // namespace cuda
}  // namespace onnxruntime
