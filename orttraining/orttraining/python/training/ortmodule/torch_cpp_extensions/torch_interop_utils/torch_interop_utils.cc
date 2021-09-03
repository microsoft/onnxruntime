// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

// In Torch forward run (e.g. THPVariable_apply), ctx of type THPFunction* (which is also a PyObject*)
// is created. The ctx is used to run user-defined forward function and backward function as the first
// parameter. The same time, a cdata of type std::shared_ptr<PyNode> is created, cdata is owned by:
//    a). forward run output tensors as grad_fn_ property. (The full hierarchy is: Tensor own 
//        shared_pointer<TensorImpl>; TensorImpl owns std::unique_ptr<AutogradMeta>; AutogradMeta
//        manages grad_/grad_fn_/grad_accumulator_. Among them, grad_fn_ is std::shared_ptr<PyNode>,
// the so called gradient function.)
//    b). the consumer operator of forward run outputs, will let its own PyNode/Node own the grad_fn_
//        (of type std::shared_ptr<PyNode>) of all inputs that require grad.
// BUT, if we run torch computation within PythonOp, b) is lost. SO, for some cases, where forward outputs
// are not used and freed before backward function runs, the grad_fn_ (std::shared_ptr<PyNode>) references
// in a) will be released. Without b)'s reference, grad_fn_ release PyNode as reference count reach 0;
// Then when PythonOpGrad runs, segment fault.
//
// So we add b)'s reference in this Pool when forward run returns; dereference from this Pool when backward
// completes, then ~PyNode() is called, which subsquently calls ~THPFunction() destorying ctx.
class PyNodeSharedPointerPool {
 public:
  static PyNodeSharedPointerPool& GetInstance() {
    static PyNodeSharedPointerPool pool;
    return pool;
  };

  void RegisterGradFunc(const size_t& ctx_address, torch::autograd::AutogradMeta* autograd_meta){
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it == grad_fns_.end(), "should not register grad_fn twice for ctx ", ctx_address);

    // Add new entry if key hasn't been registered.
    grad_fns_.emplace(ctx_address, std::move(autograd_meta->grad_fn_));
  };

  void UnRegisterGradFunc(const size_t& ctx_address){
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it != grad_fns_.end(), "fail to find grad_fn for ctx ", ctx_address);

    grad_fns_.erase(ctx_address);
  };

 private:
  PyNodeSharedPointerPool(){};
  ~PyNodeSharedPointerPool(){};

  PyNodeSharedPointerPool(const PyNodeSharedPointerPool&) = delete;
  PyNodeSharedPointerPool& operator=(const PyNodeSharedPointerPool&) = delete;
  PyNodeSharedPointerPool(PyNodeSharedPointerPool&&) = delete;
  PyNodeSharedPointerPool& operator=(PyNodeSharedPointerPool&&) = delete;

  std::unordered_map<size_t, std::shared_ptr<torch::autograd::Node>> grad_fns_;
};


void register_grad_fn(size_t ctx_address, at::Tensor target)
{
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  PyNodeSharedPointerPool::GetInstance().RegisterGradFunc(ctx_address, autograd_meta);
}

void unregister_grad_fn(size_t ctx_address)
{
  PyNodeSharedPointerPool::GetInstance().UnRegisterGradFunc(ctx_address);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_grad_fn", &register_grad_fn, "increase grad_fn shared pointer reference.");
  m.def("unregister_grad_fn", &unregister_grad_fn, "release grad_fn shared pointer referece.");
}
