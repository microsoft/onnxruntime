// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/customregistry.h"
#include "core/framework/execution_providers.h"
#include "core/framework/session_state.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
Status KernelRegistryManager::CreateKernel(const onnxruntime::Node& node,
                                           const IExecutionProvider& execution_provider,
                                           const SessionState& session_state,
                                           /*out*/ std::unique_ptr<OpKernel>& op_kernel) const {
  std::lock_guard<OrtMutex> lock(lock_);
  if (kernel_registries_.empty()) {
    return Status(ONNXRUNTIME, FAIL, "Kernel not found.");
  }

  Status status;
  for (auto& registry : kernel_registries_) {
    status = registry->CreateKernel(node,
                                    execution_provider,
                                    session_state.GetInitializedTensors(),
                                    session_state.GetMLValueNameIdxMap(),
                                    session_state.GetFuncMgr(),
                                    op_kernel);
    if (status.IsOK()) {
      return status;
    }
  }

  return status;
}

void KernelRegistryManager::RegisterKernels(const ExecutionProviders& execution_providers,
                                            KernelRegistryPriority priority) {
  for (auto& provider : execution_providers)
    RegisterKernelRegistry(provider->GetKernelRegistry(), priority);
}

void KernelRegistryManager::RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry,
                                                   KernelRegistryPriority priority) {
  std::lock_guard<OrtMutex> lock(lock_);
  if (nullptr == kernel_registry) {
    return;
  }

  if (priority == KernelRegistryPriority::HighPriority) {
    kernel_registries_.push_front(kernel_registry);
  } else {
    kernel_registries_.push_back(kernel_registry);
  }
}

Status KernelRegistryManager::SearchKernelRegistry(const onnxruntime::Node& node,
                                                   /*out*/ const KernelCreateInfo** kernel_create_info) const {
  std::lock_guard<OrtMutex> lock(lock_);
  if (kernel_registries_.empty()) {
    return Status(ONNXRUNTIME, FAIL, "Kernel def not found.");
  }

  Status status;
  for (auto& registry : kernel_registries_) {
    *kernel_create_info = registry->TryFindKernel(node, "");
    if (*kernel_create_info != nullptr) {
      return Status::OK();
    }
  }

  return Status(ONNXRUNTIME, FAIL, "Failed to find kernel for " + node.OpType());
}

}  // namespace onnxruntime
