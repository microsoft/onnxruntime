// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ctx_pool.h"
#include "custom_function_shared.h"
#include <ATen/DLConvertor.h>
#include <torch/extension.h>

/**
 * @brief Special handling for in-place reusing in forward or backward.
 * @param kernel_info kernel-specific information.
 * @param input_tensor_address_to_tensor_input_index_map
 * @param all_outputs_of_kernel_run all outputs of the MSDomain::PythonOp/PythonOpGrad.
 * @param all_outputs_to_tensor_inputs_reuse_map
 * @param raw_input_tensors_used_inplace a dict of raw input tensors marked as inplace in
            `all_outputs_to_tensor_inputs_reuse_map`, the key is the tensor input index, value is the raw input tensor.
 * @param log_prefix
 *
 *   Detection procedures:
 *   1. Detect all outputs to tensor inputs reuse mapping.
 *   2. Validate the detected inplace_map with the registered inplace_map in ORT. For the output tensor,
 *       2.0 If the reuse mapping value is the same in both inplace_map and detected inplace_map:
 *           2.0.1 Most likely, we don't need to do anything, except 2.0.2.
 *           2.0.2 Conditions:
 *               > During forward run,
 *               > The output tensor is reusing one of input tensors,
 *               > The raw input tensor to be reused given from ORT is copied to run the forward kernels
 *                   (for two possible reasons:
 *                   a. the first time forward run, all inputs will be copied to detect
 *                   `tensor_input_indices_to_save_in_ctx`;
 *                   b. for every iteration, the input needs to be cloned because it is in
 *                   `tensor_input_indices_to_save_in_ctx`).
 *
 *               In this case, need to copy the output tensor back to the raw input tensor, to make it compatible with
 *               ORT statistically planned buffer reuse.
 *       2.1 If the reuse mapping value is NOT equal in both inplace_map and detected inplace_map:
 *           2.1.1 If the detected reuse input index is -1 (e.g. there is NO buffer reuse for this output),
 *               while user specified reuse input index is NOT -1 (ORT planned the reuse), we raise an error.
 *           2.1.2 If the detected reuse input index is NOT -1 (e.g. there is buffer reuse for this output),
 *               while user specified reuse input index is -1 (ORT did not plan the reuse). We will try to clone the
 *               output tensor before returning to ORT, to align with ORT's NO Buffer reuse plan; otherwise, once the
 *               input buffer is released by ORT memory planner, the output tensor read/write will be corrupted.
 *               Raise a warning to notify users to update inplace_map explicitly for performance consideration.
 *           2.1.3 Other cases (for example user gives a wrong mapping index compared with detected ones), raise an
 *               error.
 *   3. Do copies for 2.1.2 cases.
 *   4. Do copies for 2.0.2 cases.
 */
void detect_memory_reuse_once(
    CustomFuncOpKernelInfo& kernel_info,
    const std::unordered_map<size_t, int>& input_tensor_address_to_tensor_input_index_map,
    const std::vector<py::object>& all_outputs_of_kernel_run,
    const std::vector<int64_t>& all_outputs_to_tensor_inputs_reuse_map,
    const std::unordered_map<int, at::Tensor>& raw_input_tensors_used_inplace,
    const std::string& log_prefix) {
  // Procedure 1: Detect all outputs to tensor inputs reuse mapping, according to `all_outputs_of_kernel_run` and
  // `input_tensors_of_kernel_run`.

  TORCH_CHECK(all_outputs_to_tensor_inputs_reuse_map.size() == all_outputs_of_kernel_run.size(),
              log_prefix +
                  "all_outputs_to_tensor_inputs_reuse_map and kernel run outputs sizes not expected:" +
                  std::to_string(all_outputs_to_tensor_inputs_reuse_map.size()) + " vs " +
                  std::to_string(all_outputs_of_kernel_run.size()));

  // Detect all outputs to tensor inputs reuse mapping.
  std::vector<int> detected_reuse_map(all_outputs_of_kernel_run.size(), -1);
  for (size_t output_index = 0; output_index < all_outputs_of_kernel_run.size(); ++output_index) {
    py::object arg = all_outputs_of_kernel_run[output_index];
    if (!THPVariable_Check(arg.ptr())) {
      continue;
    }
    at::Tensor t = THPVariable_Unpack(arg.ptr());
    size_t t_data_address = static_cast<size_t>(reinterpret_cast<uintptr_t>(t.data_ptr()));
    if (input_tensor_address_to_tensor_input_index_map.find(t_data_address) != input_tensor_address_to_tensor_input_index_map.end()) {
      int tensor_input_index = input_tensor_address_to_tensor_input_index_map.at(t_data_address);
      TORCH_CHECK(tensor_input_index != -1, "Reused tensor input index should not be -1");
      detected_reuse_map[output_index] = tensor_input_index;
    }
  }

  // Procedure 2: Validate the detected inplace_map with the registered inplace_map in ORT.
  // collect the output indices that need to be cloned before returned in case 2.1.2.
  for (size_t output_index = 0; output_index < all_outputs_of_kernel_run.size(); ++output_index) {
    int detected_inplace_index = detected_reuse_map[output_index];
    int inplace_index = all_outputs_to_tensor_inputs_reuse_map[output_index];

    if (inplace_index == detected_inplace_index) {
      continue;
    }

    if (raw_input_tensors_used_inplace.count(inplace_index) &&
        !raw_input_tensors_used_inplace.at(inplace_index).defined()) {
      // Use specified inplace input index, but the input tensor is None, which means the input is not
      // a tensor, so we don't do further checks.
      continue;
    }

    // If users register inplace_map (alloc planner will do buffer reuse),
    // but detected inplace_map indicates it is NO inplace reusing, we raise an error.
    if (inplace_index != -1 && detected_inplace_index == -1) {
      throw std::runtime_error(
          log_prefix + "Fatal: ONNX Op attribute 'tensor_reuse_map' indicates " +
          std::to_string(output_index) + "-th output is reusing input " +
          std::to_string(inplace_index) + ", but detected inplace_map indicates it is NOT reusing any input. " +
          "Please update inplace_map explicitly to make it consistent " +
          "to avoid undefined behavior due to ORT's memory reuse plan. " +
          +"detected reused input index: " + std::to_string(detected_inplace_index));
    }

    if (inplace_index == -1 && detected_inplace_index != -1) {
      std::cout << log_prefix << "ONNX Op attribute "
                << "'tensor_reuse_map' doesn't indicate " << std::to_string(output_index)
                << "-th output is reusing any input, "
                << "but detected inplace_map indicates it is reusing input index "
                << std::to_string(detected_inplace_index)
                << ". A clone will be done before returning to ORT, to align with ORT's NO Buffer reuse plan. "
                << "Please update inplace_map explicitly to avoid such a copy." << std::endl;

      kernel_info.output_indices_for_clone.push_back(output_index);
      continue;
    }

    throw std::runtime_error(
        log_prefix + "Fatal: ONNX Op attribute 'tensor_reuse_map' indicates " +
        std::to_string(output_index) + "-th output is reusing input " + std::to_string(inplace_index) +
        " but detected inplace_map indicates it is reusing input index " +
        std::to_string(detected_inplace_index) +
        ". Please update inplace_map explicitly to avoid undefined behavior due to memory reuse.");
  }
}

void process_inplace_outputs(
    const CustomFuncOpKernelInfo& kernel_info,
    const std::string& func_name,
    const std::unordered_map<int, at::Tensor>& input_tensors_used_for_fw_run,
    const std::vector<int64_t>& all_outputs_to_tensor_inputs_reuse_map,
    const std::unordered_map<int, at::Tensor>& raw_input_tensors_used_inplace,
    bool is_backward,
    const std::string& log_prefix,
    std::vector<py::object>& all_outputs_of_kernel_run) {
  // Procedure 3: Do copies for 2.1.2 cases.
  for (const size_t& output_index : kernel_info.output_indices_for_clone) {
    at::Tensor t = THPVariable_Unpack(all_outputs_of_kernel_run[output_index].ptr());
    auto pp = py::reinterpret_steal<py::object>(THPVariable_Wrap(t.detach().clone()));
    all_outputs_of_kernel_run[output_index] = pp;
  }

  // Procedure 4: Do copies for 2.0.2 cases.
  if (!is_backward && kernel_info.safe_run_enabled) {
    for (auto& pair : raw_input_tensors_used_inplace) {
      auto raw_tensor_input_index = pair.first;
      auto raw_input_tensor = pair.second;
      // raw_input_tensor can be None for backward run, but backward won't go here.
      if (!raw_input_tensor.defined()) {
        continue;
      }

      // We did not do the check with tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty
      // because even for those tensor indices not in
      // tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty, we still need to do the
      // copy for the first-time run.
      if (raw_input_tensor.data_ptr() == input_tensors_used_for_fw_run.at(raw_tensor_input_index).data_ptr()) {
        // If the raw input tensor is not copied, we don't need this handling.
        continue;
      }

      // for each tensor, we don't do the copy once.
      bool copied = false;
      std::vector<size_t> output_indices_reusing_current_raw_input;
      for (size_t output_index = 0; output_index < all_outputs_to_tensor_inputs_reuse_map.size(); ++output_index) {
        if (all_outputs_to_tensor_inputs_reuse_map[output_index] == raw_tensor_input_index) {
          output_indices_reusing_current_raw_input.push_back(output_index);
        }
      }

      auto output_tensor_address =
          THPVariable_Unpack(all_outputs_of_kernel_run[output_indices_reusing_current_raw_input[0]].ptr()).data_ptr();
      for (size_t& output_index : output_indices_reusing_current_raw_input) {
        auto t = THPVariable_Unpack(all_outputs_of_kernel_run[output_index].ptr());
        TORCH_CHECK(output_tensor_address == t.data_ptr(),
                    "Outputs reusing the same input tensor should have the same address.");

        if (!copied) {
          // Only need a copy once.
          // Inplace copy only happens for non-leaf variables, so we have to set requires_grad to False.
          raw_input_tensor.requires_grad_(false);
          raw_input_tensor.copy_(t);

          // Comment below for debugging.
          // std::cout << "Copy output tensor " << output_index << " to raw input tensor " << raw_tensor_input_index << "."
          //           << (!kernel_info.is_first_run
          //                   ? "Provide output to input reuse mapping to avoid the copy overhead."
          //                   : "")
          //           << std::endl;
          copied = true;
        }

        all_outputs_of_kernel_run[output_index] = py::reinterpret_steal<py::object>(THPVariable_Wrap(raw_input_tensor));
      }
    }
  }
}

void dlpack_capsule_destructor(PyObject* data) {
  if (!PyCapsule_IsValid(data, "dltensor")) {
    // early out, see DLPack spec: if a consuming library sets the capsule
    // name to something else, they own it and we don't need to do anything
    return;
  }
  DLManagedTensor* dlMTensor =
      (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
}
