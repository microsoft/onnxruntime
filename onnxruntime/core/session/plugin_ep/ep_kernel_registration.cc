// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_kernel_registration.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/error_code_helper.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/tensor.h"
#include "core/session/allocator_adapters.h"
#include "core/session/plugin_ep/ep_api.h"
#include "core/providers/cpu/controlflow/utils.h"

//
// OrtSharedPrePackedWeightCache
//
OrtSharedPrePackedWeightCache::OrtSharedPrePackedWeightCache(onnxruntime::PrePackedWeights& container,
                                                             onnxruntime::AllocatorPtr allocator)
    : container_(container), allocator_(std::move(allocator)) {}

void OrtSharedPrePackedWeightCache::SetBuffers(void** data_ptrs, size_t* data_sizes, size_t num_buffers) {
  container_.buffers_.clear();
  container_.buffer_sizes_.clear();

  container_.buffers_.reserve(num_buffers);
  container_.buffer_sizes_.reserve(num_buffers);

  for (size_t i = 0; i < num_buffers; i++) {
    auto data_unique_ptr = onnxruntime::IAllocatorUniquePtr<void>(data_ptrs[i], onnxruntime::BufferDeleter(allocator_));
    container_.buffers_.push_back(std::move(data_unique_ptr));
    container_.buffer_sizes_.push_back(data_sizes[i]);
  }
}

bool OrtSharedPrePackedWeightCache::HasData() const noexcept {
  return !container_.buffers_.empty();
}

void OrtSharedPrePackedWeightCache::ReleaseAllData() noexcept {
  for (onnxruntime::IAllocatorUniquePtr<void>& data_unique_ptr : container_.buffers_) {
    data_unique_ptr.release();
  }

  container_.buffers_.clear();
  container_.buffer_sizes_.clear();
}

namespace onnxruntime {

/// <summary>
/// OpKernel that wraps a OrtKernelImpl provided by a plugin EP.
/// </summary>
class PluginEpOpKernel final : public controlflow::IControlFlowKernel {
 private:
  // Prevents calling constructor directly without having to make it private (required by std::make_unique).
  struct PrivateTag {};

 public:
  PluginEpOpKernel(const OpKernelInfo& info, PrivateTag)
      : controlflow::IControlFlowKernel{info} {}  // must use ::Create()

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

    if (kernel_impl_->ort_version_supported < 24 || kernel_impl_->PrePackWeight == nullptr) {
      // OrtKernelImpl does not define a PrePack implementation.
      is_packed = false;
      return Status::OK();
    }

    // Convert AllocatorPtr to an OrtAllocator* (that wraps the AllocatorPtr) and cache it.
    OrtAllocator* ort_allocator = GetPrePackOrtAllocator(alloc);

    // Create a non-owning OrtValue that wraps the const Tensor& with an empty deleter.
    // This is passed to OrtKernelImpl::PrePackWeight() as a const OrtValue*.
    // The above reasons make the const_cast relatively "safe".
    // Note: Documentation for OrtKernelImpl::PrePackWeight disallows caching the OrtValue pointer.
    auto empty_tensor_deleter = [](void* /*data*/) -> void { /* do not delete Tensor (not owned) */ };
    const OrtValue ort_value(const_cast<Tensor*>(&tensor), DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);

    // Only allow kernel to store/share pre-packed weights if the weight data will be stored in cpu-accessible memory.
    // ORT requires that the data reside in cpu memory to be able to compute the hash of the weight's contents.
    //
    // If the allocator does not use CPU memory, we pass a NULL OrtSharedPrePackedWeightCache instance to the kernel to
    // indicate that storing/sharing is not allowed and the kernel should manage the memory for the pre-packed weight.
    std::optional<OrtSharedPrePackedWeightCache> shared_weight_cache;

    if (prepacked_weights != nullptr && alloc->Info().device.UsesCpuMemory()) {
      ORT_RETURN_IF(!prepacked_weights->buffers_.empty() || !prepacked_weights->buffer_sizes_.empty(),
                    "PluginEpOpKernel::PrePack() expected PrePackedWeights instance to be initially empty");
      shared_weight_cache.emplace(OrtSharedPrePackedWeightCache(*prepacked_weights, alloc));
    }

    ORT_RETURN_IF_ERROR(ToStatusAndRelease(
        kernel_impl_->PrePackWeight(kernel_impl_, &ort_value, input_idx,
                                    ort_allocator,
                                    shared_weight_cache.has_value() ? &*shared_weight_cache : nullptr,
                                    &is_packed)));

    const bool tried_to_share = shared_weight_cache.has_value() && shared_weight_cache->HasData();
    ORT_RETURN_IF(tried_to_share && !is_packed, "OrtKernelImpl::PrePackWeight() tried to share packed weight data ",
                  "but did not set the `is_packed` output parameter to true.");

    return Status::OK();
  }

  Status UseSharedPrePackedBuffers_V2(std::vector<BufferUniquePtr>& buffer_unique_ptrs,
                                      gsl::span<const size_t> buffer_sizes,
                                      int input_idx, /*out*/ bool& used_shared_buffers) override {
    assert(kernel_impl_ != nullptr);  // Should be ensured by PluginEpOpKernel::Create().

    if (kernel_impl_->ort_version_supported < 24 || kernel_impl_->SetSharedPrePackedWeight == nullptr) {
      // OrtKernelImpl does not define an implementation. The session state, which calls this function,
      // generates an error if necessary (i.e., kernel indicated it wanted to share weights but did not define this).
      used_shared_buffers = false;
      return Status::OK();
    }

    std::vector<const void*> buffer_data_ptrs;

    buffer_data_ptrs.reserve(buffer_unique_ptrs.size());
    std::transform(buffer_unique_ptrs.begin(), buffer_unique_ptrs.end(), std::back_inserter(buffer_data_ptrs),
                   [](const BufferUniquePtr& buff) -> const void* { return buff.get(); });

    ORT_RETURN_IF_ERROR(ToStatusAndRelease(
        kernel_impl_->SetSharedPrePackedWeight(kernel_impl_, buffer_data_ptrs.data(), buffer_sizes.data(),
                                               buffer_data_ptrs.size(), input_idx)));

    used_shared_buffers = true;
    return Status::OK();
  }

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override {
    assert(kernel_impl_ != nullptr);  // Should be ensured by PluginEpOpKernel::Create().

    if (kernel_impl_->ort_version_supported < 24 || kernel_impl_->SetupSubgraphExecutionInfo == nullptr) {
      // OrtKernelImpl does not define an implementation. This is an error because this function will only
      // be called by ORT when necessary (for controlflow ops).
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "PluginEpOpKernel instance is missing an implementation of ",
                             "OrtKernelImpl::SetupSubgraphExecutionInfo");
    }

    const auto* ort_state = reinterpret_cast<const OrtSessionState*>(&session_state);
    const auto* ort_subgraph_state = reinterpret_cast<const OrtSessionState*>(&subgraph_session_state);

    return ToStatusAndRelease(
        kernel_impl_->SetupSubgraphExecutionInfo(kernel_impl_, ort_state, attribute_name.c_str(),
                                                 ort_subgraph_state));
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
    for (auto& ort_allocator_wrapper : prepack_ort_allocators_) {
      if (ort_allocator_wrapper->GetWrappedIAllocator().get() == i_allocator) {
        return ort_allocator_wrapper.get();
      }
    }

    // Create a new OrtAllocatorImplWrappingIAllocator
    auto ort_allocator_wrapper = std::make_unique<OrtAllocatorImplWrappingIAllocator>(std::move(alloc));

    prepack_ort_allocators_.push_back(std::move(ort_allocator_wrapper));
    return prepack_ort_allocators_.back().get();
  }

  OrtKernelImpl* kernel_impl_ = nullptr;

  // We create and cache a OrtAllocator that wraps each unique IAllocator passed to PrePack(). Need to keep these
  // OrtAllocator instances alive because the plugin EP kernel implementation uses the OrtAllocators to allocate
  // and free packed weight data. Note: use a vector instead of an unordered_map because this will almost always
  // contain only one element and we want to limit the size of this class.
  std::vector<std::unique_ptr<OrtAllocatorImplWrappingIAllocator>> prepack_ort_allocators_;
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

//
// PluginEpIfKernel
//

/*static*/
Status PluginEpIfKernel::Create(const OpKernelInfo& info, /*out*/ std::unique_ptr<PluginEpIfKernel>& out) {
  out = std::make_unique<PluginEpIfKernel>(info, PrivateTag{});
  return Status::OK();
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpIfKernel::ComputeImpl(OrtKernelImpl* this_ptr,
                                                      OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpIfKernel*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpIfKernel::SetupSubgraphExecutionInfoImpl(
    OrtKernelImpl* this_ptr,
    const OrtSessionState* ort_session_state,
    const char* attribute_name_c_str,
    const OrtSessionState* ort_subgraph_session_state) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpIfKernel*>(this_ptr);

  const auto& session_state = *reinterpret_cast<const SessionState*>(ort_session_state);
  const auto& subgraph_session_state = *reinterpret_cast<const SessionState*>(ort_subgraph_session_state);
  const std::string attribute_name(attribute_name_c_str);

  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.SetupSubgraphExecutionInfo(session_state,
                                                                                       attribute_name,
                                                                                       subgraph_session_state));
  return nullptr;
}

/*static*/
void ORT_API_CALL PluginEpIfKernel::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpIfKernel*>(this_ptr);
}

PluginEpIfKernel::PluginEpIfKernel(const OpKernelInfo& info, PrivateTag)
    : OrtKernelImpl{},
      kernel_(info) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  SetupSubgraphExecutionInfo = SetupSubgraphExecutionInfoImpl;
}

//
// PluginEpLoopKernel
//

/*static*/
Status PluginEpLoopKernel::Create(const OpKernelInfo& info, const OrtLoopKernelConfig& config,
                                  /*out*/ std::unique_ptr<PluginEpLoopKernel>& out) {
  auto concat_output_func =
      [ep_concat_func = config.concat_output_func,
       ep_concat_func_state = config.concat_output_func_state](void* stream,
                                                               std::vector<OrtValue>& per_iteration_output,
                                                               void* output,
                                                               size_t output_size_in_bytes) -> Status {
    std::vector<OrtValue*> value_ptrs;

    value_ptrs.reserve(per_iteration_output.size());
    std::transform(per_iteration_output.begin(), per_iteration_output.end(), std::back_inserter(value_ptrs),
                   [](OrtValue& value) -> OrtValue* { return &value; });

    return ToStatusAndRelease(ep_concat_func(ep_concat_func_state, stream,
                                             value_ptrs.data(), value_ptrs.size(),
                                             output, output_size_in_bytes));
  };

  out = std::make_unique<PluginEpLoopKernel>(info, concat_output_func, PrivateTag{});
  return Status::OK();
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpLoopKernel::ComputeImpl(OrtKernelImpl* this_ptr,
                                                        OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpLoopKernel*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
void ORT_API_CALL PluginEpLoopKernel::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpLoopKernel*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpLoopKernel::SetupSubgraphExecutionInfoImpl(
    OrtKernelImpl* this_ptr,
    const OrtSessionState* ort_session_state,
    const char* attribute_name_c_str,
    const OrtSessionState* ort_subgraph_session_state) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpLoopKernel*>(this_ptr);

  const auto& session_state = *reinterpret_cast<const SessionState*>(ort_session_state);
  const auto& subgraph_session_state = *reinterpret_cast<const SessionState*>(ort_subgraph_session_state);
  const std::string attribute_name(attribute_name_c_str);

  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.SetupSubgraphExecutionInfo(session_state,
                                                                                       attribute_name,
                                                                                       subgraph_session_state));
  return nullptr;
}

PluginEpLoopKernel::PluginEpLoopKernel(const OpKernelInfo& info, Loop::ConcatOutput concat_output_func, PrivateTag)
    : OrtKernelImpl{},
      kernel_(info) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  SetupSubgraphExecutionInfo = SetupSubgraphExecutionInfoImpl;
  kernel_.SetConcatOutputFunc(concat_output_func);
}

//
// PluginEpScanKernel
//

/*static*/
template <>
Status PluginEpScanKernel<8>::Create(const OpKernelInfo& info, const OrtScanKernelConfig& config,
                                     /*out*/ std::unique_ptr<PluginEpScanKernel<8>>& out) {
  if (config.zero_data_func == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Kernel configuration for Scan opset 8 requires a valid ",
                           "OrtScanZeroDataFunc function.");
  }

  if (config.transpose_func != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "A kernel configuration with a OrtScanTransposeFunc ",
                           "function is invalid for Scan opset 8");
  }

  auto zero_data_func =
      [ep_func = config.zero_data_func,
       ep_func_state = config.zero_data_func_state](void* data, size_t size_in_bytes) -> Status {
    return ToStatusAndRelease(ep_func(ep_func_state, data, size_in_bytes));
  };

  scan::detail::DeviceHelpers device_helpers{};
  device_helpers.set_data_to_zero_func = zero_data_func;

  out = std::make_unique<PluginEpScanKernel<8>>(info, device_helpers, PrivateTag{});
  return Status::OK();
}

/*static*/
template <>
Status PluginEpScanKernel<9>::Create(const OpKernelInfo& info, const OrtScanKernelConfig& config,
                                     /*out*/ std::unique_ptr<PluginEpScanKernel<9>>& out) {
  if (config.transpose_func == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Kernel configuration for Scan opset >= 9 requires a valid ",
                           "OrtScanTransposeFunc function.");
  }

  if (config.zero_data_func != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "A kernel configuration with a OrtScanZeroDataFunc ",
                           "function is invalid for Scan opset >= 9");
  }

  auto transpose_func =
      [ep_func = config.transpose_func,
       ep_func_state = config.transpose_func_state](const gsl::span<const size_t>& permutations,
                                                    const Tensor& input, Tensor& output, Stream* stream) -> Status {
    auto empty_tensor_deleter = [](void* /*data*/) -> void { /* do not delete Tensor (not owned) */ };
    const OrtValue ort_value_input(const_cast<Tensor*>(&input), DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);
    OrtValue ort_value_output(&output, DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);
    OrtSyncStream* ort_stream = reinterpret_cast<OrtSyncStream*>(stream);

    return ToStatusAndRelease(ep_func(ep_func_state, permutations.data(), permutations.size(),
                                      &ort_value_input, ort_stream, &ort_value_output));
  };

  scan::detail::DeviceHelpers device_helpers{};
  device_helpers.transpose_func = transpose_func;

  out = std::make_unique<PluginEpScanKernel<9>>(info, device_helpers, PrivateTag{});
  return Status::OK();
}

/*static*/
template <>
OrtStatus* ORT_API_CALL PluginEpScanKernel<8>::ComputeImpl(OrtKernelImpl* this_ptr,
                                                           OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpScanKernel<8>*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
template <>
OrtStatus* ORT_API_CALL PluginEpScanKernel<9>::ComputeImpl(OrtKernelImpl* this_ptr,
                                                           OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpScanKernel<9>*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
template <>
void ORT_API_CALL PluginEpScanKernel<8>::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpScanKernel<8>*>(this_ptr);
}

/*static*/
template <>
void ORT_API_CALL PluginEpScanKernel<9>::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpScanKernel<9>*>(this_ptr);
}

/*static*/
template <>
OrtStatus* ORT_API_CALL PluginEpScanKernel<8>::SetupSubgraphExecutionInfoImpl(
    OrtKernelImpl* this_ptr,
    const OrtSessionState* ort_session_state,
    const char* attribute_name_c_str,
    const OrtSessionState* ort_subgraph_session_state) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpScanKernel<8>*>(this_ptr);

  const auto& session_state = *reinterpret_cast<const SessionState*>(ort_session_state);
  const auto& subgraph_session_state = *reinterpret_cast<const SessionState*>(ort_subgraph_session_state);
  const std::string attribute_name(attribute_name_c_str);

  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.SetupSubgraphExecutionInfo(session_state,
                                                                                       attribute_name,
                                                                                       subgraph_session_state));
  return nullptr;
}

/*static*/
template <>
OrtStatus* ORT_API_CALL PluginEpScanKernel<9>::SetupSubgraphExecutionInfoImpl(
    OrtKernelImpl* this_ptr,
    const OrtSessionState* ort_session_state,
    const char* attribute_name_c_str,
    const OrtSessionState* ort_subgraph_session_state) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpScanKernel<9>*>(this_ptr);

  const auto& session_state = *reinterpret_cast<const SessionState*>(ort_session_state);
  const auto& subgraph_session_state = *reinterpret_cast<const SessionState*>(ort_subgraph_session_state);
  const std::string attribute_name(attribute_name_c_str);

  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.SetupSubgraphExecutionInfo(session_state,
                                                                                       attribute_name,
                                                                                       subgraph_session_state));
  return nullptr;
}

template <>
PluginEpScanKernel<8>::PluginEpScanKernel(const OpKernelInfo& info,
                                          const scan::detail::DeviceHelpers& device_helpers, PrivateTag)
    : OrtKernelImpl{},
      kernel_(info) {
  ort_version_supported = ORT_API_VERSION;
  Compute = PluginEpScanKernel<8>::ComputeImpl;
  Release = PluginEpScanKernel<8>::ReleaseImpl;
  SetupSubgraphExecutionInfo = PluginEpScanKernel<8>::SetupSubgraphExecutionInfoImpl;
  kernel_.SetDeviceHelpers(device_helpers);
}

template <>
PluginEpScanKernel<9>::PluginEpScanKernel(const OpKernelInfo& info,
                                          const scan::detail::DeviceHelpers& device_helpers, PrivateTag)
    : OrtKernelImpl{},
      kernel_(info) {
  ort_version_supported = ORT_API_VERSION;
  Compute = PluginEpScanKernel<9>::ComputeImpl;
  Release = PluginEpScanKernel<9>::ReleaseImpl;
  SetupSubgraphExecutionInfo = PluginEpScanKernel<9>::SetupSubgraphExecutionInfoImpl;
  kernel_.SetDeviceHelpers(device_helpers);
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
