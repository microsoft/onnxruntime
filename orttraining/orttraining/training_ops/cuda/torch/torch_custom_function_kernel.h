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
  PythonOp(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
    inplace_ = info.GetAttrOrDefault("inplace", static_cast<int64_t>(0));
    ORT_THROW_IF_ERROR(info.GetAttr("call_convention", &call_convention_));

    // Input tensors.
    input_tensor_types_ = info.GetAttrsOrDefault("input_tensor_types", std::vector<int64_t>());
    input_tensor_requires_grads_ = info.GetAttrsOrDefault("input_tensor_requires_grads", std::vector<int64_t>());

    ORT_ENFORCE(input_tensor_types_.size() == Node().InputDefs().size());

    // Input int scalars.
    input_int_scalars_ = info.GetAttrsOrDefault("input_int_scalars", std::vector<int64_t>());
    input_int_scalar_positions_ = info.GetAttrsOrDefault("input_int_scalar_positions", std::vector<int64_t>());

    ORT_ENFORCE(input_int_scalars_.size() == input_int_scalar_positions_.size());

    // Input float scalars.
    input_float_scalars_ = info.GetAttrsOrDefault("input_float_scalars", std::vector<float>());
    input_float_scalar_positions_ = info.GetAttrsOrDefault("input_float_scalar_positions", std::vector<int64_t>());

    ORT_ENFORCE(input_float_scalars_.size() == input_float_scalar_positions_.size());

    // Input int tuples.
    input_int_tuples_ = info.GetAttrsOrDefault("input_int_tuples", std::vector<int64_t>());
    input_int_tuple_positions_ = info.GetAttrsOrDefault("input_int_tuple_positions", std::vector<int64_t>());
    input_int_tuple_begins_ = info.GetAttrsOrDefault("input_int_tuple_begins", std::vector<int64_t>());

    ORT_ENFORCE(input_int_tuple_positions_.size() == input_int_tuple_begins_.size());

    // Input float tuples.
    input_float_tuples_ = info.GetAttrsOrDefault("input_float_tuples", std::vector<float>());
    input_float_tuple_positions_ = info.GetAttrsOrDefault("input_float_tuple_positions", std::vector<int64_t>());
    input_float_tuple_begins_ = info.GetAttrsOrDefault("input_float_tuple_begins", std::vector<int64_t>());

    ORT_ENFORCE(input_float_tuple_positions_.size() == input_float_tuple_begins_.size());

    input_pointer_scalars_ = info.GetAttrsOrDefault("input_pointer_scalars", std::vector<int64_t>());
    input_pointer_scalar_positions_ = info.GetAttrsOrDefault("input_pointer_scalar_positions", std::vector<int64_t>());

    ORT_ENFORCE(input_pointer_scalars_.size() == input_pointer_scalar_positions_.size());

    // Output tensors.
    output_tensor_types_ = info.GetAttrsOrDefault("output_tensor_types", std::vector<int64_t>());
    output_tensor_requires_grads_ = info.GetAttrsOrDefault("output_tensor_requires_grads", std::vector<int64_t>());

    create_const_args();
    create_arg_positions();
  };

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void add_int_scalar_args();
  void add_input_tuple_args();
  void add_float_tuple_args();
  void add_pointer_scalar_args();
  void create_const_args();
  void create_arg_positions();

  std::vector<int64_t> const_arg_positions_;
  std::vector<void*> const_args_;
  std::vector<int64_t> arg_positions_;

  // Name of containing class. For example, MyReLU.
  std::string name_;
  int64_t inplace_;
  std::string call_convention_;

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
  PythonOpGrad(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
    ORT_THROW_IF_ERROR(info.GetAttrs("input_tensor_types", input_tensor_types_));
    ORT_THROW_IF_ERROR(info.GetAttrs("output_tensor_types", output_tensor_types_));
    input_tensor_requires_grads_ = info.GetAttrsOrDefault("input_tensor_requires_grads", std::vector<int64_t>());
    output_tensor_requires_grads_ = info.GetAttrsOrDefault("output_tensor_requires_grads", std::vector<int64_t>());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // Name of containing class. For example, MyReLU.
  std::string name_;
  // Input types of MyReLU.backward(...).
  std::vector<int64_t> input_tensor_types_;
  // Output types of MyReLU.apply(...).
  std::vector<int64_t> output_tensor_types_;
  std::vector<int64_t> input_tensor_requires_grads_;
  std::vector<int64_t> output_tensor_requires_grads_;
};

}  // namespace cuda
}  // namespace onnxruntime
