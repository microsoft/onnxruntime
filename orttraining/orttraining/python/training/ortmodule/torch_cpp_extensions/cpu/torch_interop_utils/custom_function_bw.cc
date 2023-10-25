// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ctx_pool.h"
#include "custom_function_shared.h"
#include "custom_function_bw.h"

#include <ATen/DLConvertor.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/extension.h>

#ifdef NVTX3_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif

std::vector<PyObject*> custom_function_backward_runner(const char* func_name_char,
                                                       void* callback,
                                                       const std::vector<int64_t>& requires_grad_flags,
                                                       const std::vector<int64_t>& tensor_type_flags,
                                                       const bool is_training_mode,
                                                       const std::vector<int64_t>& inplace_map,
                                                       const char* kernel_invoke_id_char,
                                                       const bool safe_run_mode_enabled,
                                                       const std::vector<PyObject*>& args) {
  pybind11::gil_scoped_acquire gil;

  try {
    std::string func_name(func_name_char);
    std::string kernel_invoke_id(kernel_invoke_id_char);
    bool is_backward = true;
    std::string log_prefix = func_name + " -> " + (is_backward ? "Backward " : "Forward ");

    at::AutoGradMode enable_grad(false);
    auto it = KernelInfoStore::GetInstance().GetKernelInfoMap().find(kernel_invoke_id);
    if (it == KernelInfoStore::GetInstance().GetKernelInfoMap().end()) {
      KernelInfoStore::GetInstance().GetKernelInfoMap().emplace(
          kernel_invoke_id,
          CustomFuncOpKernelInfo(kernel_invoke_id, safe_run_mode_enabled));
    }

    CustomFuncOpKernelInfo& kernel_info = KernelInfoStore::GetInstance().GetKernelInfoMap().at(kernel_invoke_id);

    std::unordered_map<int, at::Tensor> raw_input_tensors_used_inplace;
    std::unordered_map<int, at::Tensor> input_tensors_used_for_bw_run;

    int tensor_input_index = 0;
    std::vector<py::object> raii_call_args;
    raii_call_args.reserve(args.size());
    py::object ctx = py::reinterpret_borrow<py::object>(args[0]);
    raii_call_args.push_back(ctx);
    for (size_t arg_index = 1; arg_index < args.size(); ++arg_index) {
      if (tensor_type_flags[arg_index] != 1) {
        raii_call_args.push_back(py::reinterpret_borrow<py::object>(args[arg_index]));
        continue;
      }

      at::Tensor tensor;
      bool is_dlpack = PyCapsule_IsValid(args[arg_index], "dltensor") != 0;
      if (is_dlpack) {
        tensor = torch::utils::tensor_fromDLPack(args[arg_index]);
      } else {
        TORCH_CHECK(args[arg_index] == Py_None, "Only None is supported for non-tensor input.");
        PyObject* fw_kernel_invoke_id = PyObject_GetAttrString(ctx.ptr(), "fw_kernel_invoke_id");
        std::string fw_kernel_invoke_id_str =
            py::cast<std::string>(py::reinterpret_borrow<py::object>(fw_kernel_invoke_id));
        CustomFuncOpKernelInfo& fw_kernel_info =
            KernelInfoStore::GetInstance().GetKernelInfoMap().at(fw_kernel_invoke_id_str);
        if (fw_kernel_info.materialize_grads) {
          auto& config = fw_kernel_info.materialize_grads_config.at(arg_index - 1);
          tensor = at::zeros(std::get<0>(config), std::get<1>(config));  // shift by 1 to skip context input.
        }
      }

      if (kernel_info.safe_run_enabled) {
        bool is_input_used_inplace = std::find(inplace_map.begin(), inplace_map.end(), arg_index) !=
                                     inplace_map.end();
        if (is_input_used_inplace) {
          raw_input_tensors_used_inplace[tensor_input_index] = tensor;
        }
        input_tensors_used_for_bw_run[tensor_input_index] = tensor;
      }

      if (tensor.defined()) {
        raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor)));
      } else {
        raii_call_args.push_back(py::none());
      }

      tensor_input_index++;
    }

    py::tuple call_args = py::cast(raii_call_args);
    py::object ret;
    {
      at::AutoGradMode enable_grad(false);
      ret = py::reinterpret_steal<py::object>(PyObject_CallObject(reinterpret_cast<PyObject*>(callback),
                                                                  call_args.ptr()));
    }
    if (PyErr_Occurred()) {
      PyErr_Print();
      throw std::runtime_error("Python function execution fails with the above information.");
    }

    std::vector<py::object> all_outputs_of_kernel_run;
    if (THPVariable_Check(ret.ptr())) {
      all_outputs_of_kernel_run.push_back(ret);
    } else {
      TORCH_CHECK(PyTuple_Check(ret.ptr()), "Python function must return a tuple.");
      all_outputs_of_kernel_run = ret.cast<std::vector<py::object>>();
    }

    if (kernel_info.safe_run_enabled) {
      if (kernel_info.is_first_run) {
        // key: tensor data address;
        // value: if the tensor is defined it records the tensor input index, otherwise, -1.
        std::unordered_map<size_t, int> input_tensor_address_to_tensor_input_index_map;
        input_tensor_address_to_tensor_input_index_map.reserve(input_tensors_used_for_bw_run.size());
        for (auto& input : input_tensors_used_for_bw_run) {
          if (input.second.defined()) {
            input_tensor_address_to_tensor_input_index_map.insert(
                {{static_cast<size_t>(reinterpret_cast<uintptr_t>(input.second.data_ptr())),
                  input.first + 1}}); /* skip the ctx input*/
          }
        }

        detect_memory_reuse_once(kernel_info,
                                 input_tensor_address_to_tensor_input_index_map,
                                 all_outputs_of_kernel_run /*all_outputs_of_kernel_run*/,
                                 inplace_map /*all_outputs_to_tensor_inputs_reuse_map*/,
                                 raw_input_tensors_used_inplace,
                                 log_prefix);
      }

      process_inplace_outputs(kernel_info,
                              func_name,
                              input_tensors_used_for_bw_run,
                              inplace_map /*all_outputs_to_tensor_inputs_reuse_map*/,
                              raw_input_tensors_used_inplace,
                              is_backward /*is_backward*/,
                              log_prefix,
                              all_outputs_of_kernel_run /*all_outputs_of_kernel_run*/);
    }

    unregister_grad_fn(ctx);

    std::vector<PyObject*> rets;
    for (auto& py_obj : all_outputs_of_kernel_run) {
      PyObject* obj = py_obj.ptr();

      if (!THPVariable_Check(obj)) {
        Py_INCREF(obj);
        rets.push_back(obj);
        continue;
      }

      DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(obj));
      rets.push_back(PyCapsule_New(dlMTensor, "dltensor", dlpack_capsule_destructor));
    }

    if (kernel_info.is_first_run) {
      kernel_info.is_first_run = false;
    }
    return rets;
  } catch (const std::exception& e) {
    std::cerr << "custom_function_backward_runner failed with " << e.what() << std::endl;
    throw;
  }
}
