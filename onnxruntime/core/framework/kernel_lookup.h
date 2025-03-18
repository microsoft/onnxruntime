// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <gsl/gsl>
#include "core/framework/execution_provider.h"  // for IExecutionProvider::IKernelLookup
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"

namespace onnxruntime {

/**
 * Utility class for performing kernel lookup.
 * Primary usage pattern is to be created during graph partitioning and passed to IExecutionProvider::GetCapability().
 */
class KernelLookup final : public IExecutionProvider::IKernelLookup {
 public:
  KernelLookup(ProviderType provider_type,
               gsl::span<const gsl::not_null<const KernelRegistry*>> kernel_registries,
               const IKernelTypeStrResolver& kernel_type_str_resolver,
               const logging::Logger& logger)
      : provider_type_{provider_type},
        kernel_registries_{kernel_registries},
        kernel_type_str_resolver_{kernel_type_str_resolver},
        logger_{logger} {
    ORT_ENFORCE(!provider_type_.empty(), "provider_type must be specified.");
  }

  const KernelCreateInfo* LookUpKernel(const Node& node) const override {
    const KernelCreateInfo* kernel_create_info{};
    for (const auto& registry : kernel_registries_) {
      const auto lookup_status = registry->TryFindKernel(node, provider_type_, kernel_type_str_resolver_, logger_,
                                                         &kernel_create_info);
      if (lookup_status.IsOK() && kernel_create_info != nullptr) {
        return kernel_create_info;
      }
    }

    return nullptr;
  }

 private:
  ProviderType provider_type_;
  const gsl::span<const gsl::not_null<const KernelRegistry*>> kernel_registries_;
  const IKernelTypeStrResolver& kernel_type_str_resolver_;
  const logging::Logger& logger_;
};

}  // namespace onnxruntime
