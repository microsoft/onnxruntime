// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/python_function.h>

// In PyTorch forward run (e.g. THPFunction_apply), ctx of type THPFunction* (which is also a PyObject*)
// is created (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/python_function.cpp#L673).
// The ctx is used to run user-defined forward function and backward function as the first
// parameter. The same time, a cdata of type std::shared_ptr<PyNode> is created
// (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/python_function.cpp#L677),
// cdata is owned by:
//    a). forward run output tensors as grad_fn_ property. (The full hierarchy is: Tensor owns
//        shared_pointer<TensorImpl>; TensorImpl owns std::unique_ptr<AutogradMeta>; AutogradMeta
//        manages grad_/grad_fn_/grad_accumulator_. Among them, grad_fn_ is std::shared_ptr<PyNode>,
//        e.g, the so called gradient function.)
//        https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/variable.h#L194
//    b). the consumer operator of forward run outputs, will let its own PyNode/Node (gradient function)
//        owns the grad_fn_ (of type std::shared_ptr<PyNode>) of all inputs that require grad.
//        https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/function.h#L263
// BUT, if we run torch computation within PythonOp, b) is lost. So for some cases, where forward outputs
// are not used and freed before backward function runs, the grad_fn_ (std::shared_ptr<PyNode>) references
// in a) will be released. Without b)'s reference, grad_fn_ release PyNode as reference count reach 0;
// Then when PythonOpGrad runs, segment fault.
//
// So we add b)'s reference in this Pool when forward run returns; dereference from this Pool when backward
// completes, then ~PyNode() is called, which subsequently calls ~THPFunction() destroying ctx.
class PyNodeSharedPointerPool {
 public:
  static PyNodeSharedPointerPool& GetInstance() {
    static PyNodeSharedPointerPool pool;
    return pool;
  };

  void RegisterGradFuncAndRemoveFromAutoGrad(const size_t& ctx_address,
                                             torch::autograd::AutogradMeta* autograd_meta) {
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it == grad_fns_.end(), "should not register grad_fn twice for ctx ", ctx_address);

    // Add new entry if key hasn't been registered.
    // After this, the grad_fn_ is removed from torch autograd.
    grad_fns_.emplace(ctx_address, std::move(autograd_meta->grad_fn_));
    TORCH_CHECK(autograd_meta->grad_fn_ == nullptr, "fail to remove grad_fn_ from torch autograd for ctx ",
                ctx_address);
  };

  void UnRegisterGradFunc(const size_t& ctx_address) {
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it != grad_fns_.end(), "fail to find grad_fn for ctx ", ctx_address);

    grad_fns_.erase(ctx_address);
  };

  void ClearAll() {
    grad_fns_.clear();
  }

 private:
  PyNodeSharedPointerPool(){};
  ~PyNodeSharedPointerPool(){};

  PyNodeSharedPointerPool(const PyNodeSharedPointerPool&) = delete;
  PyNodeSharedPointerPool& operator=(const PyNodeSharedPointerPool&) = delete;
  PyNodeSharedPointerPool(PyNodeSharedPointerPool&&) = delete;
  PyNodeSharedPointerPool& operator=(PyNodeSharedPointerPool&&) = delete;

  std::unordered_map<size_t, std::shared_ptr<torch::autograd::Node>> grad_fns_;
};

void clear_grad_fns_for_next_edges(at::Tensor target, std::vector<at::Tensor> saved_tensors) {
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

void register_grad_fn_and_remove_from_autograd(size_t ctx_address, at::Tensor target) {
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  PyNodeSharedPointerPool::GetInstance().RegisterGradFuncAndRemoveFromAutoGrad(ctx_address, autograd_meta);
}

void unregister_grad_fn(size_t ctx_address) {
  PyNodeSharedPointerPool::GetInstance().UnRegisterGradFunc(ctx_address);
}

// Supposed to be cleared on python program exit to resolve the following issue:
// When training program exits, PyNodeSharedPointerPool destructor is called, if grad_fns_ is not empty,
// PyNode::release_variables() will be called.
// (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/python_function.cpp#L168)
// On The other hand, there is a known issue when acquiring GIL in pybind11 destructors, there will be
// probably a deadlock issue. (https://github.com/pybind/pybind11/issues/1446)
// The resolution here, we remove all maintained states before the program exits.

// A known existing issue: when forward functions are called repeatedly without corresponding backward calls,
// grad functions keep accumulating without releasing, there might be memory (bound to those gradient functions) leaks.
// Ideally this usually won't happen in real training cases, so it should be fine.

// We CANNOT explicitly clear grad functions before each forward pass to mitigate the known issue above.
// For example:
//     loss1 = forward_run(inputs1)
//     loss2 = forward_run(inputs2)
//     loss = loss1 + loss2
//     loss.backward()
// If we clear grad functions at the beginning of the second `forward_run`, when `loss.backward()` runs,
// the backward path of `loss1` will fail to run PythonOpGrad ops (if there is any).
void clear_all_grad_fns() {
  PyNodeSharedPointerPool::GetInstance().ClearAll();
}

bool get_materialize_grads(at::Tensor target) {
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  const auto& grad_fn = autograd_meta->grad_fn_;
  auto py_node_fn = dynamic_cast<torch::autograd::PyNode*>(grad_fn.get());
  TORCH_CHECK(py_node_fn != nullptr, "grad_fn is not PyNode type.");
  THPFunction* py_fn = (THPFunction*)py_node_fn->obj;
  return py_fn->materialize_grads;
}

std::vector<bool> are_tensors_marked_as_dirty(at::Tensor target, std::vector<at::Tensor> tensors_to_check) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_grad_fn_and_remove_from_autograd", &register_grad_fn_and_remove_from_autograd,
        "Increase grad_fn shared pointer reference.");
  m.def("unregister_grad_fn", &unregister_grad_fn, "Release grad_fn shared pointer reference.");
  m.def("clear_all_grad_fns", &clear_all_grad_fns, "Clear all grad_fn shared pointer references.");
  m.def("clear_grad_fns_for_next_edges", &clear_grad_fns_for_next_edges,
        "Remove reference on next edges' gradient functions.");
  m.def("get_materialize_grads", &get_materialize_grads, "Return whether materialize_grads is enabled or not.");
  m.def("are_tensors_marked_as_dirty", &are_tensors_marked_as_dirty, "Return whether the tensors are marked dirty or not.");
}
