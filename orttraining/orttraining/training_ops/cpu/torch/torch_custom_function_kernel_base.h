// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

std::vector<OrtValue*> CreateOrtValueArgs(OpKernelContext* context,
                                          const size_t begin_index,
                                          const size_t num_arg);

class PythonOpBase {
 protected:
  PythonOpBase(const OpKernelInfo& info);

  void AddIntScalarArgs();
  void AddInputTupleArgs();
  void AddFloatTupleArgs();
  void AddPointerScalarArgs();
  void CreateConstArgs();
  void CreateArgPositions();

  void SetContextOutput(OpKernelContext* context, void* diff_ctx) const;
  void SetOtherOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_args) const;

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

class PythonOpGradBase {
 protected:
  PythonOpGradBase(const OpKernelInfo& info);

  void SetPositions();
  void SetOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_args) const;

  // Name of containing class. For example, MyReLU.
  std::string name_;
  int64_t inplace_;
  // Input types of MyReLU.backward(...).
  std::vector<int64_t> input_tensor_types_;
  // Output types of MyReLU.apply(...).
  std::vector<int64_t> output_tensor_types_;
  std::vector<int64_t> input_tensor_requires_grads_;
  std::vector<int64_t> output_tensor_requires_grads_;
  std::vector<int64_t> arg_positions_;
  std::vector<int64_t> const_arg_positions_;
};

}  // namespace contrib
}  // namespace onnxruntime
