// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include <memory>
#include <variant>
#include <unordered_map>

#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/graph/graph_viewer.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
struct KernelCreateInfo;
class ExecutionProviders;
class IExecutionProvider;
class KernelRegistry;
class OpKernel;
class SessionState;

// Kernel registries' manager.
// There're 2 kinds of kernel registries with priority from high to low as below,
// 1. Custom execution provider type specific kernel registries.
// 2. common execution provider type specific kernel registries.
// The 1st and 2nd ones are shared across sessions.

// This class is not thread safe.
class KernelRegistryManager {
 public:
  KernelRegistryManager() = default;

  // Register kernels from providers
  Status RegisterKernels(const ExecutionProviders& execution_providers);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // The registry passed in this function has highest priority than anything already in this KernelRegistryManager,
  // and anything registered from RegisterKernels
  // For example, if you do:
  // RegisterKernels(providers)
  // RegisterKernelRegistry(A);
  // RegisterKernelRegistry(B);
  // Then B > A > providers
  void RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry);
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

  /**
   * Gets kernel registries for the specified provider type.
   * @param provider_type provider type string
   * @return The kernel registries. This also includes custom registries. These may contain kernels that don't belong
   *         to this provider. The caller should do the filtering.
   */
  InlinedVector<gsl::not_null<const KernelRegistry*>> GetKernelRegistriesByProviderType(
      const std::string& provider_type) const {
    InlinedVector<gsl::not_null<const KernelRegistry*>> result;
    result.reserve(custom_kernel_registries_.size() + 1);
    for (auto& registry : custom_kernel_registries_) {
      result.push_back(registry.get());
    }
    auto iter = provider_type_to_registry_.find(provider_type);
    if (iter != provider_type_to_registry_.end()) result.push_back(iter->second.get());
    return result;
  }

  // This function assumes the node is already assigned to an execution provider
  // Don't call this function before graph partition is done
  Status SearchKernelRegistry(const Node& node,
                              /*out*/ const KernelCreateInfo** kernel_create_info) const;

  /**
   * Whether this node can be run on this provider
   */
  static bool HasImplementationOf(const KernelRegistryManager& r, const Node& node, const std::string& provider_type);

  Status CreateKernel(const Node& node,
                      const IExecutionProvider& execution_provider,
                      SessionState& session_state,
                      const KernelCreateInfo& kernel_create_info, std::unique_ptr<OpKernel>& out) const;

  const IKernelTypeStrResolver& GetKernelTypeStrResolver() const {
    return std::visit([](auto&& r) -> const IKernelTypeStrResolver& { return r; }, kernel_type_str_resolver_variant_);
  }

  void SetKernelTypeStrResolver(KernelTypeStrResolver&& kernel_type_str_resolver) {
    kernel_type_str_resolver_variant_ = std::move(kernel_type_str_resolver);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelRegistryManager);

 private:
  // key is provider type. Each kernel registry in this collection only belongs to one specific provider
  std::unordered_map<std::string, std::shared_ptr<KernelRegistry>> provider_type_to_registry_;
  // Each kernel registry may contain kernels from many different providers.
  // in order to search kernels from a specific provider, we have to iterate all its elements
  std::list<std::shared_ptr<KernelRegistry>> custom_kernel_registries_;

  // kernel type str resolver used by kernel registries for kernel matching
  using KernelTypeStrResolverVariant = std::variant<
#if !defined(ORT_MINIMAL_BUILD)
      OpSchemaKernelTypeStrResolver,  // the default in a full build
#endif
      KernelTypeStrResolver  // the default in a minimal build
      >;
  KernelTypeStrResolverVariant kernel_type_str_resolver_variant_;
};
}  // namespace onnxruntime
