// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_kernel_registration.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>

#include "core/framework/error_code_helper.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/tensor.h"
#include "core/session/allocator_adapters.h"
#include "core/session/plugin_ep/ep_api.h"

namespace onnxruntime {

/// <summary>
/// OpKernel that wraps a OrtKernelImpl provided by a plugin EP.
/// </summary>
class PluginEpOpKernel final : public OpKernel {
 private:
  // Prevents calling constructor directly without having to make it private (required by make_unique_ptr).
  struct PrivateTag {};

  // Stores a mapping between a IAllocator* and the OrtAllocator* that wraps it.
  struct PrePackAllocatorMapping {
    PrePackAllocatorMapping(IAllocator* i_alloc, std::unique_ptr<OrtAllocatorImplWrappingIAllocator> ort_alloc)
        : i_allocator(i_alloc), ort_allocator(std::move(ort_alloc)) {}

    IAllocator* i_allocator{};                                            // Original IAllocator passed to PrePack()
    std::unique_ptr<OrtAllocatorImplWrappingIAllocator> ort_allocator{};  // Wrapper over IAllocator
  };

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

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) override {
    assert(kernel_impl_ != nullptr);  // Should be ensured by PluginEpOpKernel::Create().

    if (kernel_impl_->PrePackWeight == nullptr) {
      // OrtKernelImpl did not define a PrePack implementation.
      is_packed = false;
      return Status::OK();
    }

    // Only allow kernel to store/share pre-packed weights if the weight data will be stored in cpu-accessible memory.
    // ORT requires that the data reside in cpu memory to be able to compute the hash of the weight's contents.
    //
    // If the allocator does not use CPU memory, we pass a NULL OrtPrePackWeightCache instance to the kernel to indicate
    // that storing/sharing is not allowed and the kernel should manage the memory for the pre-packed weight.
    bool enable_weight_sharing = alloc->Info().device.UsesCpuMemory() && prepacked_weights != nullptr;
    OrtSharedPrePackedWeightCache shared_weight_cache = {};

    // Convert AllocatorPtr to an OrtAllocator* and cache it to ensure it lives long enough.
    OrtAllocator* ort_allocator = GetPrePackOrtAllocator(alloc);

    // Create a non-owning OrtValue that wraps the const Tensor& with an empty deleter.
    // This is passed to OrtKernelImpl::PrePackConstantTensor() as a const OrtValue*.
    // The above reasons make the const_cast relatively "safe".
    // Note: Documentation for OrtKernelImpl::PrePackConstantTensor disallows caching the OrtValue pointer.
    auto empty_tensor_deleter = [](void* /*data*/) -> void { /* do not delete Tensor (not owned) */ };
    const OrtValue ort_value(const_cast<Tensor*>(&tensor), DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);

    ORT_RETURN_IF_ERROR(ToStatusAndRelease(
        kernel_impl_->PrePackWeight(kernel_impl_, &ort_value, input_idx,
                                    ort_allocator, enable_weight_sharing ? &shared_weight_cache : nullptr,
                                    &is_packed)));

    if (is_packed && enable_weight_sharing) {
      ORT_RETURN_IF(shared_weight_cache.allocator != ort_allocator,
                    "OrtKernelImpl::PrePackWeight() did not allocate shared pre-packed weights with the ",
                    "required OrtAllocator.");

      for (size_t i = 0; i < shared_weight_cache.buffer_data_ptrs.size(); i++) {
        void* data_ptr = shared_weight_cache.buffer_data_ptrs[i].release();
        size_t num_bytes = shared_weight_cache.buffer_sizes[i];

        prepacked_weights->buffers_.push_back(IAllocatorUniquePtr<void>(data_ptr, BufferDeleter(alloc)));
        prepacked_weights->buffer_sizes_.push_back(num_bytes);
      }
    }

    return Status::OK();
  }

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& buffer_unique_ptrs,
                                   int input_idx, /*out*/ bool& used_shared_buffers) override {
    assert(kernel_impl_ != nullptr);  // Should be ensured by PluginEpOpKernel::Create().

    if (kernel_impl_->SetSharedPrePackedWeight == nullptr) {
      // OrtKernelImpl did not define an implementation. The session state, which calls this function,
      // generates an error if necessary (i.e., kernel indicated it wanted to share weights but did not define this).
      used_shared_buffers = false;
      return Status::OK();
    }

    std::vector<const void*> buffer_data_ptrs;

    buffer_data_ptrs.reserve(buffer_unique_ptrs.size());
    std::transform(buffer_unique_ptrs.begin(), buffer_unique_ptrs.end(), std::back_inserter(buffer_data_ptrs),
                   [](const BufferUniquePtr& buff) -> const void* { return buff.get(); });

    ORT_RETURN_IF_ERROR(ToStatusAndRelease(
        kernel_impl_->SetSharedPrePackedWeight(kernel_impl_, buffer_data_ptrs.data(), buffer_data_ptrs.size(),
                                               input_idx)));

    used_shared_buffers = true;
    return Status::OK();
  }

 private:
  /// <summary>
  /// Gets the cached OrtAllocator for the given AllocatorPtr passed to PrePack().
  /// </summary>
  /// <param name="alloc"></param>
  /// <returns></returns>
  OrtAllocator* GetPrePackOrtAllocator(AllocatorPtr alloc) {
    IAllocator* i_allocator = alloc.get();

    // Try to find an existing OrtAllocator* that wraps the given IAllocator*
    for (auto& alloc_mapping : prepack_allocator_mappings_) {
      if (alloc_mapping.i_allocator == i_allocator) {
        return alloc_mapping.ort_allocator.get();
      }
    }

    // Generate a new mapping from IAllocator* to OrtAllocator* and return the latter.
    PrePackAllocatorMapping alloc_mapping(i_allocator,
                                          std::make_unique<OrtAllocatorImplWrappingIAllocator>(std::move(alloc)));
    prepack_allocator_mappings_.push_back(std::move(alloc_mapping));

    return prepack_allocator_mappings_.back().ort_allocator.get();
  }

  OrtKernelImpl* kernel_impl_ = nullptr;

  // We create and cache a OrtAllocator for each unique IAllocator passed to PrePack(). Need to keep these
  // OrtAllocator instances alive because the plugin EP kernel implementation uses the OrtAllocators to allocate
  // and free packed weight data.
  std::vector<PrePackAllocatorMapping> prepack_allocator_mappings_;
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
