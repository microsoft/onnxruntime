// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Use void* instead of PyObject* to avoid adding unnecessary
// python.h dependency for the consumers (for example: the
// provider bridge file).
#ifdef ENABLE_TRAINING_TORCH_INTEROP

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

#include "orttraining/core/framework/torch/torch_proxy.h"

namespace onnxruntime {
namespace contrib {

class PythonOpBase {
 public:
  PythonOpBase(const OpKernelInfo& info) {
    Init(info);
  }

  void Init(const OpKernelInfo& info);

  void RunForward(OpKernelContext* context,
                  void** diff_ctx,
                  std::vector<OrtValue>& returned_ortvalues) const;

  void SetOutputs(OpKernelContext* context,
                  void* diff_ctx,
                  std::vector<OrtValue>& returned_args) const;

  ~PythonOpBase() {
    Clear();
  }

  void Clear();

 protected:
  class ConstantArgSet {
   private:
    class ConstantArg {
     public:
      ConstantArg(int64_t position, void* data_ptr, bool is_owned)
          : position(position), data_ptr(data_ptr), is_owned(is_owned) {}
      int64_t position;  // input offset in the input lists
      void* data_ptr;    // pointer to the data
      bool is_owned;     // whether the data is owned by this PythonOp kernel.
    };

   public:
    // Append new constant argument. Fail when called after Finalize() got called.
    void Add(int64_t position, void* data_ptr, bool owned) {
      ORT_ENFORCE(positions_.empty() && data_ptrs_.empty(),
                  "Cannot add constant arg after Finalize()");
      args_.emplace_back(ConstantArg(position, data_ptr, owned));
    }

    // Finalize the constant arg set. This is called after all constant args are added.
    // Fail when called more than once.
    void Finalize() {
      ORT_ENFORCE(positions_.empty() && data_ptrs_.empty());
      positions_.reserve(args_.size());
      for (auto& arg : args_) {
        positions_.push_back(arg.position);
      }

      data_ptrs_.reserve(args_.size());
      for (auto& arg : args_) {
        data_ptrs_.push_back(arg.data_ptr);
      }
    }

    size_t Size() const {
      return args_.size();
    }

    const std::vector<ConstantArg>& GetArgs() const {
      return args_;
    }

    const std::vector<int64_t>& GetPositions() const {
      return positions_;
    }

    const std::vector<void*>& GetDataPtrs() const {
      return data_ptrs_;
    }

   private:
    std::vector<ConstantArg> args_;
    std::vector<int64_t> positions_;
    std::vector<void*> data_ptrs_;
  };

  // A collection for all non-tensor input arguments, we treated them all as constants, including primitive types and
  // tuples, and also string or other user defined data types (represented in pointer in the attribute
  // "input_pointer_scalars").
  ConstantArgSet const_arg_set_;

  std::vector<int64_t> arg_positions_;

  // Name of containing class. For example, MyReLU.
  std::string name_;
  int64_t inplace_;
  std::string input_convention_;
  bool is_training_mode_;
  // input_requires_grads_[i] indicates if the i-th inputs of apply() should have gradient.
  std::vector<int64_t> input_requires_grads_;

  // Attributes of input tensors for calling MyReLU.apply(...).
  // Types. input_tensor_types_[i] is the element type of the i-th tensor.
  std::vector<int64_t> input_tensor_types_;

  // Concatenation of all bools from apply(...) 's inputs.
  std::vector<int64_t> input_bool_scalars_;
  std::vector<int64_t> input_bool_scalar_positions_;

  // Concatenation of all ints from apply(...) 's inputs.
  std::vector<int64_t> input_int_scalars_;
  std::vector<int64_t> input_int_scalar_positions_;

  // Concatenation of all floats from apply(...) 's inputs.
  std::vector<float> input_float_scalars_;
  std::vector<int64_t> input_float_scalar_positions_;

  // Concatenation of all bool tuples from apply(...) 's inputs.
  std::vector<int64_t> input_bool_tuples_;
  std::vector<int64_t> input_bool_tuple_positions_;
  std::vector<int64_t> input_bool_tuple_begins_;

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

 private:
  void AddPrimitiveTypeScalarArgs();
  void AddInputTupleArgs();
  void AddFloatTupleArgs();
  void AddPointerScalarArgs();
  void CreateConstArgs();
  void CreateArgPositions();

  void SetContextOutput(OpKernelContext* context, void* diff_ctx) const;
  void SetOtherOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_args) const;

  std::string kernel_invoke_id_;
};

class PythonOpGradBase {
 public:
  PythonOpGradBase(const OpKernelInfo& info) {
    Init(info);
  };

  void Init(const OpKernelInfo& info);

  void RunBackward(OpKernelContext* context,
                   std::vector<OrtValue>& returned_ortvalues) const;

  void SetOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_args) const;

 protected:
  // Name of containing class. For example, MyReLU.
  std::string name_;
  int64_t inplace_;
  // Input types of MyReLU.backward(...).
  std::vector<int64_t> input_tensor_types_;

  std::string output_convention_;
  // Output types of MyReLU.apply(...).
  std::vector<int64_t> output_tensor_types_;
  std::vector<int64_t> output_tensor_requires_grads_;
  std::vector<int64_t> arg_positions_;
  std::vector<int64_t> const_arg_positions_;

 private:
  void SetPositions();

  std::string kernel_invoke_id_;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif
