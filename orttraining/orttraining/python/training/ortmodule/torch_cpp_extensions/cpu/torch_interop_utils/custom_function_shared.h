// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <torch/extension.h>

// Uncomment this line to enable NVTX profiling
// #define NVTX3_ENABLED 1

class CustomFuncOpKernelInfo {
 public:
  CustomFuncOpKernelInfo(const std::string& invoke_id, bool safe_run) {
    kernel_invoke_id = invoke_id;
    safe_run_enabled = safe_run;
  }

  // kernel_invoke_id is a string contains session thread id, op kernel creation time stamp in ms, a random int,
  // and address of op_kernel pointer. This can guarantee the uniqueness of the key in case of multiple
  // instances of a same named PythonOp/PythonOpGrad in one session, or multiple sessions.
  std::string kernel_invoke_id;

  // For the tensors generated from ORT backend, there is special handling here:
  // 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
  //    all such tensors will be cloned in case they are saved in context (but ORT backend is not aware of the
  //    reference, may release the content of the tensor before it is needed in backward). Once
  //    `autograd.Function.apply` completes, by checking the existence of the tensor in the saved_tensors,
  //    `_GlobalOpKernelInfoMap` is updated to save the input indices that are saved in context.
  // 2. For the subsequent runs, if the input index is in `tensor_input_indices_to_save_in_ctx`, the tensor
  //    will be cloned before fed into `autograd.Function.apply` as input.
  std::unordered_map<int, bool> tensor_input_indices_to_save_in_ctx;

  // To align with PyTorch `ctx.set_materialize_grads(False|True)`, default to be true.
  // materialize_grads_config is a map from output index to (device, dtype, shape) of the output tensor, used
  // for materializing the gradient of the output tensor in backward.
  bool materialize_grads{true};
  // key: output index, value: (shape, tensor options including device, layerout, data types, etc)
  std::unordered_map<size_t, std::tuple<std::vector<int64_t>, c10::TensorOptions>> materialize_grads_config;

  // For the tensors generated from ORT backend, there is special handling here:
  // 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
  //    all such tensors will be cloned (with gradient) in case they are marked as dirty (if not cloned, but marked
  //    as dirty, PyTorch will complain the tensor is a leaf, should not be used for inplace update). Once
  //    `autograd.Function.apply` completes, by checking the existence of the tensor in the dirty_tensors,
  //    `_GlobalOpKernelInfoMap` is updated to save the input indices that are marked as dirty.
  // 2. For the subsequent runs, if the input index is in `tensor_input_indices_for_mark_dirty`, the tensor
  //    will be cloned (with gradient) before fed into `autograd.Function.apply` as input.
  std::unordered_map<int, bool> tensor_input_indices_for_mark_dirty;

  // A list of output indices that needs to be clone before returned, due to inplace update analysis.
  std::vector<size_t> output_indices_for_clone;

  bool is_first_run{true};
  bool safe_run_enabled{false};
};

void detect_memory_reuse_once(
    CustomFuncOpKernelInfo& kernel_info,
    const std::unordered_map<size_t, int>& input_tensor_address_to_tensor_input_index_map,
    const std::vector<py::object>& all_outputs_of_kernel_run,
    const std::vector<int64_t>& all_outputs_to_tensor_inputs_reuse_map,
    const std::unordered_map<int, at::Tensor>& raw_input_tensors_used_inplace,
    const std::string& log_prefix);

void process_inplace_outputs(
    const CustomFuncOpKernelInfo& kernel_info,
    const std::string& func_name,
    const std::unordered_map<int, at::Tensor>& input_tensors_used_for_fw_run,
    const std::vector<int64_t>& all_outputs_to_tensor_inputs_reuse_map,
    const std::unordered_map<int, at::Tensor>& raw_input_tensors_used_inplace,
    bool is_backward,
    const std::string& log_prefix,
    std::vector<py::object>& all_outputs_of_kernel_run);

void dlpack_capsule_destructor(PyObject* data);

class KernelInfoStore {
 public:
  static KernelInfoStore& GetInstance() {
    static KernelInfoStore instance;
    return instance;
  }

  std::unordered_map<std::string, CustomFuncOpKernelInfo>& GetKernelInfoMap() {
    return kernel_info_map_;
  }

 private:
  std::unordered_map<std::string, CustomFuncOpKernelInfo> kernel_info_map_;
};
