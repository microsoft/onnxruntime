// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "orttraining/core/framework/torch/python_common.h"

#ifndef SHARED_PROVIDER
#include "core/framework/ort_value.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/platform/env.h"
#endif

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

// For handling temporary PyObject pointer newly created with Py_XXX APIs, here is our practice:
// Convention:
//     Wrap those PyObject* in format of "PythonObjectPtr(Py_XXX(), PythonObjectDeleter)".
// Explaination:
//     That means, for the PyObject* created by Py_XXX(), its refcnt will be decreased by one
//     in the PythonObjectDeleter which is triggered once lifetime of PythonObjectPtr instance
//     ends.

void PythonObjectDeleter(PyObject* ptr);
using PythonObjectPtr = std::unique_ptr<PyObject, std::function<void(PyObject*)>>;

/// Use void* instead of PyObject* to avoid add unnecessary
/// python.h dependency for the consumers.
class TorchProxy {
 public:
  static TorchProxy& GetInstance() {
    static TorchProxy proxy;
    return proxy;
  };

  void Forward(
      const std::string& func_name,
      void* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<std::optional<OrtValue>>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      const bool is_training_mode,
      const std::vector<int64_t>& inplace_map,
      const std::string& invoke_id,
      bool safe_run_mode_enabled,
      void** diff_ctx,
      std::vector<OrtValue>& returned_ortvalues);

  void Backward(
      const std::string& func_name,
      void* callback,
      const std::vector<std::optional<OrtValue>>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      const std::vector<int64_t>& inplace_map,
      const std::string& invoke_id,
      bool safe_run_mode_enabled,
      std::vector<OrtValue>& returned_ortvalues);

  /**
   * @brief Run given function to get output to input reuse map.
   *
   * @param input_alias_func Python function to run.
   *  The function should take a serialized PythonOp NodeProto string as input, return a tuple of two lists.
   *  The signature of the function should be:
   *     def alias_input(node_proto_str: str):
   *         fw_alias_map = [1, -1, -1]
   *         bw_alias_map = [-1, 0]
   *         return fw_alias_map, bw_alias_map
   * @param node_proto_str The serialized PythonOp NodeProto string.
   * @param fw_output_to_input_alias_map Used as returned value, return the output to input alias map for forward pass.
   *   For example, if the inputs of the torch.autograd.Function are [non_tensor_a, tensor_b],
   *   outputs are [tensor_x, tensor_y, tensor_z], and the alias map is [1, -1, -1], this is explained as:
   *   tensor_x is reusing the input tensor_b, tensor_y and tensor_z are not reusing any input.
   *   The value of alias map is 0 based input index. -1 means the output is not reusing any input.
   * @param bw_output_to_input_alias_map Used as returned value, return the output to input alias map for backward pass.
   *   For example, if the inputs of the torch.autograd.Function are [tensor_x_grad, None, None],
   *   outputs are [None, tensor_b_grad], and the alias map is [-1, 0], this is explained as:
   *   tensor_b_grad is reusing the input tensor_x_grad.
   *   The value of alias map is 0 based grad input index. -1 means the output is not reusing any input.
   */
  void RunInputAliasFunction(
      void* input_alias_func,
      const std::string& node_proto_str,
      std::vector<int64_t>& fw_output_to_input_alias_map,
      std::vector<int64_t>& bw_output_to_input_alias_map);

 private:
  TorchProxy(){};
  ~TorchProxy(){};

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TorchProxy);

  // All member functions should be exclusively used because
  // Python has a global interpreter.
  std::mutex mutex_;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
