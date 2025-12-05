// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_kernel_registration.h"

#include <memory>
#include <utility>

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
  PluginEpOpKernel(const OpKernelInfo& info, PrivateTag) : OpKernel{info} {}  // must use ::Create()

  static Status Create(FuncManager& fn_manager, const OpKernelInfo& info,
                       OrtKernelCreateFunc kernel_create_func, void* kernel_create_func_state,
                       /*out*/ std::unique_ptr<PluginEpOpKernel>& op_kernel);

  ~PluginEpOpKernel() {
    if (kernel_impl_ != nullptr) {
      kernel_impl_->Release(kernel_impl_);
    }
  }

  Status Compute(OpKernelContext* ctx) const override {
    assert(kernel_impl_ != nullptr);  // Should be ensured by PluginEpOpKernel::Create().
    return ToStatusAndRelease(kernel_impl_->Compute(kernel_impl_, reinterpret_cast<OrtKernelContext*>(ctx)));
  }

 private:
  OrtKernelImpl* kernel_impl_ = nullptr;
};

/*static*/
Status PluginEpOpKernel::Create(FuncManager& /*fn_manager*/, const OpKernelInfo& info,
                                OrtKernelCreateFunc kernel_create_func, void* kernel_create_func_state,
                                /*out*/ std::unique_ptr<PluginEpOpKernel>& op_kernel) {
  // OpKernel's constructor *copies* the OpKernelInfo.
  // Therefore, must create the OpKernel instance immediately so that we can pass the actual OpKernelInfo
  // to the plugin EP's kernel creation function.
  op_kernel = std::make_unique<PluginEpOpKernel>(info, PrivateTag{});
  const OrtKernelInfo* kernel_info = reinterpret_cast<const OrtKernelInfo*>(&op_kernel->Info());

  ORT_RETURN_IF_ERROR(ToStatusAndRelease(
      kernel_create_func(kernel_create_func_state, kernel_info, &op_kernel->kernel_impl_)));
  ORT_RETURN_IF(op_kernel->kernel_impl_ == nullptr, "OrtKernelCreateFunc returned a NULL OrtKernelImpl");

  return Status::OK();
}

/// <summary>
/// A functor that creates a PluginEpOpKernel instance using the creation function (+ state) provided by a plugin EP.
/// </summary>
class PluginEpKernelCreateFunctor {
 public:
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
  OrtKernelCreateFunc kernel_create_func_ = nullptr;
  void* kernel_create_func_state_ = nullptr;
};

// Make a KernelCreateInfo for a plugin EP's kernel
KernelCreateInfo MakePluginEpKernelCreateInfo(const KernelDef* kernel_def,
                                              OrtKernelCreateFunc kernel_create_func,
                                              void* kernel_create_func_state) {
  auto kernel_def_copy = std::make_unique<KernelDef>(*kernel_def);
  PluginEpKernelCreateFunctor kernel_create_functor(kernel_create_func, kernel_create_func_state);
  return KernelCreateInfo(std::move(kernel_def_copy), kernel_create_functor);
}

// Copies a const OrtKernelRegistry into a shared_ptr<KernelRegistry>.
static Status CopyEpKernelRegistry(const OrtKernelRegistry* ep_registry,
                                   /*out*/ std::shared_ptr<KernelRegistry>& registry_copy) {
  if (ep_registry == nullptr) {
    registry_copy = nullptr;
    return Status::OK();
  }

  const KernelRegistry* src_registry = reinterpret_cast<const KernelRegistry*>(ep_registry);
  auto dst_registry = std::make_shared<KernelRegistry>();

  for (const auto& [key, src_create_info] : src_registry->GetKernelCreateMap()) {
    auto dst_kernel_def = std::make_unique<KernelDef>(*src_create_info.kernel_def);
    KernelCreateInfo dst_create_info(std::move(dst_kernel_def), src_create_info.kernel_create_func);

    ORT_RETURN_IF_ERROR(dst_registry->Register(std::move(dst_create_info)));
  }

  registry_copy = std::move(dst_registry);
  return Status::OK();
}

// Gets an OrtEp instance's kernel registry.
Status GetPluginEpKernelRegistry(OrtEp& ort_ep, /*out*/ std::shared_ptr<KernelRegistry>& kernel_registry) {
  kernel_registry = nullptr;

  if (ort_ep.ort_version_supported < 24) {
    // OrtEp::GetKernelRegistry was added in ORT 1.24.0, but this OrtEp uses an older ORT version.
    return Status::OK();
  }

  if (ort_ep.GetKernelRegistry != nullptr) {
    const OrtKernelRegistry* ep_registry = nullptr;

    ORT_RETURN_IF_ERROR(ToStatusAndRelease(ort_ep.GetKernelRegistry(&ort_ep, &ep_registry)));

    // ORT needs a shared_ptr<KernelRegistry> due to the IExecutionProvider::GetKernelRegistry() interface.
    // We copy the EP's OrtKernelRegistry into a new shared_ptr<KernelRegistry> to ensure the EP fully owns
    // the lifetime of the registry it created.
    ORT_RETURN_IF_ERROR(CopyEpKernelRegistry(ep_registry, kernel_registry));
  }

  return Status::OK();
}

}  // namespace onnxruntime
