// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_kernel_registration.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/kernel_registry.h"
#include "core/session/plugin_ep/ep_api.h"

namespace onnxruntime {

/// <summary>
/// OpKernel that wraps a OrtKernelImpl provided by a plugin EP.
/// </summary>
class PluginEpOpKernel final : public OpKernel {
 private:
  struct PrivateTag {};

 public:
  PluginEpOpKernel(const OpKernelInfo& info, PrivateTag)
      : OpKernel{info} {}

  static Status Create(FuncManager& fn_manager, const OpKernelInfo& info,
                       OrtKernelCreateFunc kernel_create_func, void* kernel_create_func_state,
                       /*out*/ std::unique_ptr<PluginEpOpKernel>& op_kernel);

  ~PluginEpOpKernel() {
    kernel_impl_->Release(kernel_impl_);
  }

  Status Compute(OpKernelContext* ctx) const override {
    return ToStatusAndRelease(kernel_impl_->Compute(kernel_impl_, reinterpret_cast<OrtKernelContext*>(ctx)));
  }

 private:
  OrtKernelImpl* kernel_impl_ = nullptr;
};

/*static*/
Status PluginEpOpKernel::Create(FuncManager& fn_manager, const OpKernelInfo& info,
                                OrtKernelCreateFunc kernel_create_func, void* kernel_create_func_state,
                                /*out*/ std::unique_ptr<PluginEpOpKernel>& op_kernel) {
  // OpKernel's constructor *copies* the OpKernelInfo.
  // Therefore, must create the OpKernel instance immediately so that we can pass the actual OpKernelInfo
  // to the plugin EP's kernel creation function.
  op_kernel = std::make_unique<PluginEpOpKernel>(info, PrivateTag{});

  OrtKernelCreateContext* create_ctx = reinterpret_cast<OrtKernelCreateContext*>(&fn_manager);
  const OrtKernelInfo* kernel_info = reinterpret_cast<const OrtKernelInfo*>(&op_kernel->Info());

  ORT_RETURN_IF_ERROR(ToStatusAndRelease(
      kernel_create_func(create_ctx, kernel_create_func_state, kernel_info, &op_kernel->kernel_impl_)));

  return Status::OK();
}

/// <summary>
/// A functor that creates a PluginEpOpKernel instance using the creation function (+ state) provided by a plugin EP.
/// </summary>
class PluginEpKernelCreateFunctor {
 public:
  PluginEpKernelCreateFunctor() : kernel_create_func_(nullptr), kernel_create_func_state_(nullptr) {}
  PluginEpKernelCreateFunctor(OrtKernelCreateFunc create_func, void* state)
      : kernel_create_func_{create_func}, kernel_create_func_state_{state} {}

  Status operator()(FuncManager& fn_manager, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
    if (kernel_create_func_ == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "PluginEpKernelCreateFunctor does not wrap a valid OrtKernelCreateFunc");
    }

    std::unique_ptr<PluginEpOpKernel> plugin_ep_op_kernel;
    ORT_RETURN_IF_ERROR(PluginEpOpKernel::Create(fn_manager, info, kernel_create_func_, kernel_create_func_state_,
                                                 plugin_ep_op_kernel));

    out = std::move(plugin_ep_op_kernel);
    return Status::OK();
  }

 private:
  OrtKernelCreateFunc kernel_create_func_;
  void* kernel_create_func_state_;
};

// Make a KernelCreateInfo for a plugin EP's kernel
KernelCreateInfo MakePluginEpKernelCreateInfo(const KernelDef* kernel_def,
                                              OrtKernelCreateFunc kernel_create_func,
                                              void* kernel_create_func_state) {
  auto kernel_def_copy = std::make_unique<onnxruntime::KernelDef>(*kernel_def);
  PluginEpKernelCreateFunctor kernel_create_functor(kernel_create_func, kernel_create_func_state);
  return KernelCreateInfo(std::move(kernel_def_copy), kernel_create_functor);
}

// Gets an OrtEp instance's kernel registry.
Status GetPluginEpKernelRegistry(OrtEp& ort_ep, /*out*/ std::shared_ptr<KernelRegistry>& kernel_registry) {
  kernel_registry = nullptr;

  if (ort_ep.GetKernelRegistry == nullptr) {
    return Status::OK();
  }

  const OrtKernelRegistry* ep_kernel_registry = nullptr;
  ORT_RETURN_IF_ERROR(ToStatusAndRelease(ort_ep.GetKernelRegistry(&ort_ep, &ep_kernel_registry)));

  if (ep_kernel_registry == nullptr) {
    return Status::OK();
  }

  kernel_registry = ep_kernel_registry->registry;
  return Status::OK();
}

}  // namespace onnxruntime
