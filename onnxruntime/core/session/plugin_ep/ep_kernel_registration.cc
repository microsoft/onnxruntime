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

Status InitKernelRegistry(OrtEp& ort_ep, /*out*/ std::shared_ptr<KernelRegistry>& kernel_registry) {
  kernel_registry = nullptr;

  if (ort_ep.GetNumKernelCreateInfos == nullptr || ort_ep.GetKernelCreateInfos == nullptr) {
    return Status::OK();
  }

  size_t num_kernels = ort_ep.GetNumKernelCreateInfos(&ort_ep);
  if (num_kernels == 0) {
    return Status::OK();
  }

  std::vector<OrtKernelCreateInfo*> kernel_create_infos(num_kernels);
  Status status = ToStatusAndRelease(ort_ep.GetKernelCreateInfos(&ort_ep, kernel_create_infos.data(), num_kernels));

  // Store OrtKernelCreateInfo provided by the plugin EP in std::unique_ptr so that they are always properly released.
  std::vector<std::unique_ptr<OrtKernelCreateInfo, decltype(&OrtExecutionProviderApi::ReleaseKernelCreateInfo)>>
      kernel_create_infos_holder;

  kernel_create_infos_holder.reserve(num_kernels);
  for (OrtKernelCreateInfo* kernel_create_info : kernel_create_infos) {
    auto holder = std::unique_ptr<OrtKernelCreateInfo, decltype(&OrtExecutionProviderApi::ReleaseKernelCreateInfo)>(
        kernel_create_info, OrtExecutionProviderApi::ReleaseKernelCreateInfo);
    kernel_create_infos_holder.push_back(std::move(holder));
  }

  if (!status.IsOK()) {
    return status;
  }

  // Add all KernelCreateInfo instances to the KernelRegistry
  kernel_registry = std::make_shared<KernelRegistry>();

  for (size_t i = 0; i < num_kernels; i++) {
    OrtKernelCreateInfo* ort_kernel_create_info = kernel_create_infos[i];
    PluginEpKernelCreateFunctor kernel_create_functor(ort_kernel_create_info->kernel_create_func,
                                                      ort_kernel_create_info->kernel_create_func_state);
    KernelCreateInfo kernel_create_info(std::make_unique<KernelDef>(ort_kernel_create_info->kernel_def),  // copy
                                        kernel_create_functor);

    ORT_RETURN_IF_ERROR(kernel_registry->Register(std::move(kernel_create_info)));
  }

  return Status::OK();
}

}  // namespace onnxruntime
