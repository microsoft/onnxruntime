// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ctx_pool.h"
#include "custom_function_shared.h"
#include "custom_function_fw.h"
#include <ATen/DLConvertor.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/autograd/python_cpp_function.h>

#ifdef NVTX3_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif

static void clear_grad_fns_for_next_edges(at::Tensor& target,
                                          std::vector<at::Tensor>& saved_tensors) {
  // For leaf tensor, there will be a AccumulateGrad (gradient function) created, which owns a
  // reference to the tensor.
  // For any user saved tensors (with save_for_backward), if the tensor is leaf, we put the map
  // {AccumulateGrad*, Tensor*} into grad_fn_to_tensor_map.
  std::unordered_map<torch::autograd::Node*, at::Tensor*> grad_fn_to_tensor_map;
  for (auto& t : saved_tensors) {
    auto grad_fn = t.grad_fn();
    if (!grad_fn) {
      grad_fn = torch::autograd::impl::try_get_grad_accumulator(t);
      if (grad_fn) {
        TORCH_CHECK(grad_fn_to_tensor_map.find(grad_fn.get()) == grad_fn_to_tensor_map.end(),
                    "found AccumulateGrad* is used by more than one tensors.");
        grad_fn_to_tensor_map.insert({grad_fn.get(), &t});
      }
    }
  }

  const auto& gradient_func_sptr = target.grad_fn();
  for (auto& edge : gradient_func_sptr->next_edges()) {
    torch::autograd::Node* node_func = edge.function.get();
    // If we find the next gradient function is AccumulateGrad, we will check whether its owned
    // tensors is in ctx.save_tensors or not. If yes, we skip it; otherwise, we clean the edge, which
    // will release the AccumulateGrad function.
    if (dynamic_cast<torch::autograd::AccumulateGrad*>(node_func)) {
      if (grad_fn_to_tensor_map.find(node_func) != grad_fn_to_tensor_map.end()) {
        // skip the edges that connect to saved_tensors. Because when unpack ctx.saved_tensors using
        // following code in backward:
        //     input, = ctx.saved_tensors
        // there is such a check: if the saved tensor is a leaf and requires grad, it should have grad accumulator.
        // If we clean the edge, then an exception "RuntimeError: No grad accumulator for a saved leaf!" will be thrown
        continue;
      } else {
        edge.function.reset();
      }
    }
  }
}

static std::vector<bool> are_tensors_marked_as_dirty(at::Tensor& target,
                                                     std::vector<at::Tensor>& tensors_to_check) {
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  const auto& grad_fn = autograd_meta->grad_fn_;
  auto py_node_fn = dynamic_cast<torch::autograd::PyNode*>(grad_fn.get());
  TORCH_CHECK(py_node_fn != nullptr, "grad_fn is not PyNode type.");
  THPFunction* py_fn = (THPFunction*)py_node_fn->obj;
  std::vector<bool> are_tensors_marked_dirty(tensors_to_check.size(), false);
  if (!py_fn->dirty_tensors)
    return are_tensors_marked_dirty;

  Py_ssize_t num_dirty = PyTuple_GET_SIZE(py_fn->dirty_tensors);
  for (const auto j : c10::irange(tensors_to_check.size())) {
    bool is_tensor_marked_dirty = false;
    for (const auto i : c10::irange(num_dirty)) {
      PyObject* obj = PyTuple_GET_ITEM(py_fn->dirty_tensors, i);
      const auto& tensor = THPVariable_Unpack(obj);
      if (tensor.is_same(tensors_to_check[j])) {
        is_tensor_marked_dirty = true;
        break;
      }
    }

    are_tensors_marked_dirty[j] = is_tensor_marked_dirty;
  }

  return are_tensors_marked_dirty;
}

std::optional<at::Tensor> try_to_get_tensor_owning_context(const py::tuple& forward_output_tensors) {
  py::object ctx = py::none();
  std::optional<at::Tensor> first_tensor_output;

  for (size_t i = 0; i < forward_output_tensors.size(); ++i) {
    PyObject* obj = forward_output_tensors[i].ptr();
    if (!THPVariable_Check(obj)) {
      continue;
    }

    at::Tensor t = THPVariable_Unpack(obj);
    if (!t.grad_fn()) {
      continue;
    }

    // Be noted, in Python, we need additional check as below.
    // For the following case, it is possible grad_fn exists, but its value is None,
    // so we need to continue to search for the first tensor having a non-None grad_fn.
    //
    //  >>> w = torch.randn(5, 6)
    //  >>> hasattr(w, "grad_fn")
    //  True
    //  >>> w.grad_fn is None
    //  True
    //  >>> w, ... = CustomFunc.apply(w) # where CustomFunc forward just return w and other tensors.
    //
    //  Then hasattr(w, "grad_fn") is True, but w.grad_fn is None.

    first_tensor_output = t;
    break;
  }

  return first_tensor_output;
}

void get_materialize_grads_once(const py::tuple& forward_output_tensors,
                                bool need_materialize_grads,
                                CustomFuncOpKernelInfo& kernel_info) {
  kernel_info.materialize_grads = need_materialize_grads;
  if (need_materialize_grads) {
    for (size_t i = 0; i < forward_output_tensors.size(); ++i) {
      PyObject* obj = forward_output_tensors[i].ptr();
      if (!THPVariable_Check(obj)) {
        continue;
      }
      at::Tensor t = THPVariable_Unpack(obj);
      kernel_info.materialize_grads_config.insert({i, {t.sizes().vec(), t.options()}});
    }

    static std::once_flag log_warning;
    std::call_once(log_warning, []() {
      std::cerr << "First-time run initialize kernel info including materialize_grads and materialize_grads_config."
                << std::endl;
    });
  }
}

py::object finalize_training_mode_forward(
    const std::unordered_map<int, at::Tensor>& input_tensors_used_for_fw_run,
    const py::tuple& forward_output_tensors,
    CustomFuncOpKernelInfo& kernel_info) {
  std::optional<at::Tensor> tensor_owning_ctx = try_to_get_tensor_owning_context(forward_output_tensors);

  if (!tensor_owning_ctx.has_value()) {
    // ctx being None in training mode means the forward function is not differentiable, so backward is not needed.
    return py::none();
  }

  const std::shared_ptr<torch::autograd::Node>& cdata = tensor_owning_ctx.value().grad_fn();
  auto py_node_fn = dynamic_cast<torch::autograd::PyNode*>(cdata.get());
  TORCH_CHECK(py_node_fn != nullptr, "cdata is not PyNode type.");

  // ret is THPFunction
  THPFunction* py_fn = (THPFunction*)py_node_fn->obj;
  py::object ret = py::reinterpret_steal<py::object>(torch::autograd::functionToPyObject(cdata));

  TORCH_CHECK(py_fn != nullptr, "cdata is not THPFunction type.");

  // The way we find saved tensor is aligned with
  // "THPFunction_saved_tensors" and "unpack_saved_variables" in PyTorch.
  std::vector<at::Tensor> saved_tensors;
  int num_saved = py_fn->saved_variables.size();
  auto saved_for = py_fn->cdata.lock();
  TORCH_INTERNAL_ASSERT(saved_for);

  for (const auto i : c10::irange(num_saved)) {
    auto unpacked_var = py_fn->saved_variables[i].unpack(saved_for);
    if (unpacked_var.defined()) {
      // TODO(pengwa): is it possible we do the copy on demand here instead of do blind
      // copy and do detection at the first iteration.
      saved_tensors.push_back(unpacked_var);
    }
  }

  if (kernel_info.is_first_run) {
    get_materialize_grads_once(forward_output_tensors, py_fn->materialize_grads, kernel_info);

    if (kernel_info.safe_run_enabled) {
      for (auto& pair : input_tensors_used_for_fw_run) {
        auto& tensor = pair.second;
        bool found = false;
        for (auto& t : saved_tensors) {
          if (t.is_same(tensor)) {
            found = true;
            break;
          }
        }
        kernel_info.tensor_input_indices_to_save_in_ctx[pair.first] = found;
      }

      // Check tensors generated by ORT are marked as dirty(for inplace update) or not .
      // If yes, save the input index of the tensor in the KernelInfoStore::GetInstance().GetKernelInfoMap().
      std::vector<at::Tensor> tensors_to_check;
      tensors_to_check.reserve(input_tensors_used_for_fw_run.size());
      for (auto& pair : input_tensors_used_for_fw_run) {
        tensors_to_check.push_back(pair.second);
      }

      std::vector<bool> are_dirty = are_tensors_marked_as_dirty(tensor_owning_ctx.value(), tensors_to_check);
      size_t index = 0;
      for (auto& pair : input_tensors_used_for_fw_run) {
        kernel_info.tensor_input_indices_for_mark_dirty[pair.first] = are_dirty[index];

        index += 1;
      }

      static std::once_flag log_warning;
      std::call_once(log_warning, []() {
        std::cerr << "First time run initialize kernel info including saved_for_forward, and mark_dirty infos." << std::endl;
      });
    }
  }

  // #FORWARD BACKWARD FUNCTION CONNECTIONS
  // #input_1(leaf, constructed by from_dlpack) < -- --reference-- --AccumulateGrad gradient function
  // #             ↓                                                                 ↑
  // #autograd.Function apply()-- -- -- -- -- --> autograd.Function backward()
  // #             ↓ |                            ↑
  // #output_1, output_2-- - shared_ptr < PyNode> -- -                            ↑
  // #             ↓ previous gradient function

  // #We remove the edges starting between current autograd.Function's gradient function and
  // #it 's input' s gradient function(e.g.AccumulateGrad gradient function), then
  // #AccumulateGrad gradient function will be destroyed, releasing the reference to input_1
  // #(https: //github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/functions/accumulate_grad.cpp#L21).
  // #The next edges are stored in Node, with which we can get next gradient function.
  // #https:  // github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/function.h#L527

  clear_grad_fns_for_next_edges(tensor_owning_ctx.value(), saved_tensors);

  // This is mainly to hold grad_fn references by registering it into our PyNodeSharedPointerPool.
  register_grad_fn_and_remove_from_autograd(ret, tensor_owning_ctx.value());

  return ret;
}

static py::object get_mockup_context_class() {
  static py::object kclass_obj;

  if (!kclass_obj.ptr()) {
    // Load the module object
    auto module =
        py::reinterpret_steal<py::object>(
            PyImport_ImportModule("onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.torch_interop_utils.fake_ctx"));
    if (!module.ptr()) {
      PyErr_Print();
      throw std::runtime_error("Fails to import the module.");
    }

    auto python_class = PyObject_FastGetAttrString(module.ptr(), "FakeContext");
    if (!PyCallable_Check(python_class.ptr())) {
      throw std::runtime_error("Cannot instantiate the Python class");
    }

    kclass_obj = py::reinterpret_borrow<py::object>(python_class.ptr());
  }

  return kclass_obj;
}

std::vector<PyObject*> custom_function_forward_runner(const char* func_name_char,
                                                      void* callback,
                                                      const std::vector<int64_t>& requires_grad_flags,
                                                      const std::vector<int64_t>& tensor_type_flags,
                                                      const bool is_training_mode,
                                                      const std::vector<int64_t>& inplace_map,
                                                      const char* kernel_invoke_id_char,
                                                      const bool safe_run_mode_enabled,
                                                      const std::vector<PyObject*>& args) {
  try {
    pybind11::gil_scoped_acquire gil;

    std::string func_name(func_name_char);
    std::string kernel_invoke_id(kernel_invoke_id_char);
    bool is_backward = false;
    std::string log_prefix = func_name + " -> " + (is_backward ? "Backward " : "Forward ");

#ifdef NVTX3_ENABLED
    nvtxRangePushA(std::string(func_name + ".fw").c_str());
#endif

    auto it = KernelInfoStore::GetInstance().GetKernelInfoMap().find(kernel_invoke_id);
    if (it == KernelInfoStore::GetInstance().GetKernelInfoMap().end()) {
      KernelInfoStore::GetInstance().GetKernelInfoMap().emplace(
          kernel_invoke_id,
          CustomFuncOpKernelInfo(kernel_invoke_id, safe_run_mode_enabled));
    }

    CustomFuncOpKernelInfo& kernel_info = KernelInfoStore::GetInstance().GetKernelInfoMap().at(kernel_invoke_id);

    std::unordered_map<int, at::Tensor> raw_input_tensors_used_inplace;
    std::unordered_map<int, at::Tensor> input_tensors_used_for_fw_run;

    int tensor_input_index = 0;
    std::vector<py::object> raii_call_args;
    if (kernel_info.safe_run_enabled) {
      raii_call_args.reserve(args.size());
    } else {
      auto python_class = get_mockup_context_class();
      // Creates an instance of the class
      PyObject* object = PyObject_CallObject(python_class.ptr(), nullptr);
      raii_call_args.reserve(args.size() + 1);
      raii_call_args.push_back(py::reinterpret_steal<py::object>(object));
    }

    for (size_t arg_index = 0; arg_index < args.size(); ++arg_index) {
      bool is_tensor = (tensor_type_flags[arg_index] == 1);
      if (!is_tensor) {
        raii_call_args.push_back(py::reinterpret_borrow<py::object>(args[arg_index]));
        continue;
      }

      // Assume it's a DLPack tensor and convert it to PyTorch tensor.
      TORCH_CHECK(PyCapsule_IsValid(args[arg_index], "dltensor") != 0, "found invalid pycapsule");
      at::Tensor tensor = torch::utils::tensor_fromDLPack(args[arg_index]);
      bool requires_grad = requires_grad_flags[arg_index] && is_training_mode;
      tensor.requires_grad_(requires_grad);

      if (kernel_info.safe_run_enabled) {
        bool is_input_used_inplace = (std::find(inplace_map.begin(), inplace_map.end(), tensor_input_index) !=
                                      inplace_map.end());
        if (is_input_used_inplace) {
          raw_input_tensors_used_inplace[tensor_input_index] = tensor;
        }

        if (kernel_info.is_first_run) {
          at::Tensor tensor_clone;
          if (is_training_mode) {
            at::AutoGradMode enable_grad(true);
            tensor_clone = tensor.clone();
            tensor_clone.requires_grad_(requires_grad);
          } else {
            tensor_clone = tensor;
          }

          raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor_clone)));
          input_tensors_used_for_fw_run[tensor_input_index] = tensor_clone;
        } else {
          // Saving tensor for backward only affect the training.
          bool is_input_index_saved_in_ctx =
              is_training_mode && kernel_info.tensor_input_indices_to_save_in_ctx.at(tensor_input_index);

          bool is_input_index_marked_dirty =
              kernel_info.tensor_input_indices_for_mark_dirty.at(tensor_input_index);

          if (is_input_index_saved_in_ctx || is_input_index_marked_dirty) {
            at::AutoGradMode enable_grad(is_input_index_marked_dirty);
            auto wrapped_arg = tensor.clone();
            wrapped_arg.requires_grad_(requires_grad);
            raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(wrapped_arg)));
            input_tensors_used_for_fw_run[tensor_input_index] = wrapped_arg;
          } else {
            raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor)));
            input_tensors_used_for_fw_run[tensor_input_index] = tensor;
          }
        }
      } else {
        raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor)));
      }

      tensor_input_index++;
    }

    if (kernel_info.safe_run_enabled && kernel_info.is_first_run) {
      // Initialize some kernel info for the first run.
      for (const auto i : c10::irange(input_tensors_used_for_fw_run.size())) {
        kernel_info.tensor_input_indices_to_save_in_ctx.insert({{i, false}});
        kernel_info.tensor_input_indices_for_mark_dirty.insert({{i, false}});
      }
    }

#ifdef NVTX3_ENABLED
    nvtxRangePushA(std::string(func_name + ".call_func").c_str());
#endif

    py::tuple call_args = py::cast(raii_call_args);
    PyObject* result_pyobj;
    {
      at::AutoGradMode enable_grad(is_training_mode && kernel_info.safe_run_enabled);
      result_pyobj = PyObject_CallObject(reinterpret_cast<PyObject*>(callback), call_args.ptr());
    }

#ifdef NVTX3_ENABLED
    nvtxRangePop();
#endif

    if (PyErr_Occurred()) {
      PyErr_Print();
    }

    if (!result_pyobj) {
      throw std::runtime_error("Get null result");
    }

    py::object ret = py::reinterpret_steal<py::object>(result_pyobj);

    py::tuple forward_outputs;
    if (THPVariable_Check(ret.ptr())) {  // Don't check be tensor?
      forward_outputs = py::make_tuple(ret);
    } else {
      TORCH_CHECK(PyTuple_Check(ret.ptr()), "Python function must return a tuple.");
      forward_outputs = ret.cast<py::tuple>();
    }

    py::object ctx;
    if (is_training_mode) {
#ifdef NVTX3_ENABLED
      std::string tag3 = func_name + ".ctx";
      nvtxRangePushA(tag3.c_str());
#endif
      if (kernel_info.safe_run_enabled) {
        ctx = finalize_training_mode_forward(input_tensors_used_for_fw_run, forward_outputs, kernel_info);
        if (!ctx.is_none()) {
          PyObject_SetAttrString(ctx.ptr(), "fw_kernel_invoke_id", py::cast(kernel_invoke_id).ptr());
        }
      } else {
        if (kernel_info.is_first_run) {
          bool need_materialize_grads = true;
          get_materialize_grads_once(forward_outputs, need_materialize_grads, kernel_info);
        }

        ctx = call_args[0];
        PyObject_SetAttrString(ctx.ptr(), "fw_kernel_invoke_id", py::cast(kernel_invoke_id).ptr());
      }

#ifdef NVTX3_ENABLED
      nvtxRangePop();
#endif
    } else {
      ctx = py::none();
    }

    std::vector<py::object> all_outputs_of_kernel_run;
    all_outputs_of_kernel_run.reserve(forward_outputs.size() + 1);
    all_outputs_of_kernel_run.push_back(ctx);
    for (size_t i = 0; i < forward_outputs.size(); ++i) {
      all_outputs_of_kernel_run.push_back(forward_outputs[i]);
    }

    if (kernel_info.safe_run_enabled) {
      if (kernel_info.is_first_run) {
        // key: tensor data address;
        // value: if the tensor is defined it records the tensor input index, otherwise, -1.
        std::unordered_map<size_t, int> input_tensor_address_to_tensor_input_index_map;
        input_tensor_address_to_tensor_input_index_map.reserve(input_tensors_used_for_fw_run.size());
        for (auto& input : input_tensors_used_for_fw_run) {
          if (input.second.defined()) {
            input_tensor_address_to_tensor_input_index_map.insert(
                {{static_cast<size_t>(reinterpret_cast<uintptr_t>(input.second.data_ptr())), input.first}});
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
                              input_tensors_used_for_fw_run,
                              inplace_map /*all_outputs_to_tensor_inputs_reuse_map*/,
                              raw_input_tensors_used_inplace,
                              false /*is_backward*/,
                              log_prefix,
                              all_outputs_of_kernel_run /*all_outputs_of_kernel_run*/);
    }

#ifdef NVTX3_ENABLED
    nvtxRangePushA(std::string(func_name + ".final").c_str());
#endif

    std::vector<PyObject*> rets;
    rets.reserve(all_outputs_of_kernel_run.size());
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

#ifdef NVTX3_ENABLED
    nvtxRangePop();
#endif

    if (kernel_info.is_first_run) {
      kernel_info.is_first_run = false;
    }

#ifdef NVTX3_ENABLED
    nvtxRangePop();
#endif

    return rets;
  } catch (const std::exception& e) {
    std::cerr << "custom_function_forward_runner failed with " << e.what() << std::endl;
    throw;
  }
}
