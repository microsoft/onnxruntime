// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ctx_pool.h"
#include <torch/extension.h>

void register_grad_fn_and_remove_from_autograd(py::object ctx, at::Tensor target) {
  uint32_t y = reinterpret_cast<uintptr_t>(ctx.ptr());
  size_t ctx_address = static_cast<size_t>(y);

  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  PyNodeSharedPointerPool::GetInstance().RegisterGradFuncAndRemoveFromAutoGrad(ctx_address, autograd_meta);
}

void unregister_grad_fn(py::object ctx) {
  uint32_t y = reinterpret_cast<uintptr_t>(ctx.ptr());
  size_t ctx_address = static_cast<size_t>(y);
  PyNodeSharedPointerPool::GetInstance().UnRegisterGradFunc(ctx_address);
}

void clear_all_grad_fns() {
  PyNodeSharedPointerPool::GetInstance().ClearAll();
}
