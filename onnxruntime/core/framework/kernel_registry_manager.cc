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
  const std::string& ptype = node.GetExecutionProviderType();
  if (ptype.empty()) {
    return Status(ONNXRUNTIME, FAIL, "The node is not placed on any Execution Provider, therefore, can't find a suitable kernel for it");
  }
  Status status;
  {
    std::lock_guard<OrtMutex> lock(lock_);
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
  }

  KernelRegistry* p = nullptr;
  {
    std::lock_guard<OrtMutex> lock(lock_);
    auto iter = provider_type_to_registry_.find(ptype);
    if (iter != provider_type_to_registry_.end()) p = iter->second.get();
  }
  if (p != nullptr) {
    status = p->CreateKernel(node,
                             execution_provider,
                             session_state.GetInitializedTensors(),
                             session_state.GetMLValueNameIdxMap(),
                             session_state.GetFuncMgr(),
                             op_kernel);
    if (status.IsOK()) {
      return status;
    }
  }

  std::ostringstream errormsg;
  errormsg << "Failed to find kernel for " << node.OpType();
  if (node.Op() != nullptr)
    errormsg << "(" << node.Op()->since_version() << ")";
  if (!node.Name().empty()) errormsg << " (node " << node.Name() << ")";
  return Status(ONNXRUNTIME, FAIL, errormsg.str());
}

void KernelRegistryManager::RegisterKernels(const ExecutionProviders& execution_providers) {
  for (auto& provider : execution_providers) {
    std::lock_guard<OrtMutex> lock(lock_);
    auto iter = provider_type_to_registry_.find(provider->Type());
    ORT_ENFORCE(iter == provider_type_to_registry_.end());
    provider_type_to_registry_.insert(std::make_pair(provider->Type(), provider->GetKernelRegistry()));
  }
}

void KernelRegistryManager::RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry) {
  if (nullptr == kernel_registry) {
    return;
  }
  std::lock_guard<OrtMutex> lock(lock_);
  kernel_registries_.push_front(kernel_registry);
}

Status KernelRegistryManager::SearchKernelRegistry(const onnxruntime::Node& node,
                                                   /*out*/ const KernelCreateInfo** kernel_create_info) const {
  const std::string& ptype = node.GetExecutionProviderType();
  if (ptype.empty()) {
    return Status(ONNXRUNTIME, FAIL, "The node is not placed on any Execution Provider");
  }
  Status status;
  {
    std::lock_guard<OrtMutex> lock(lock_);
    for (auto& registry : kernel_registries_) {
      *kernel_create_info = registry->TryFindKernel(node, "");  //the last argument is ignored
      if (*kernel_create_info != nullptr) {
        return Status::OK();
      }
    }
  }

  KernelRegistry* p = nullptr;
  {
    std::lock_guard<OrtMutex> lock(lock_);
    auto iter = provider_type_to_registry_.find(ptype);
    if (iter != provider_type_to_registry_.end()) p = iter->second.get();
  }
  if (p != nullptr) {
    *kernel_create_info = p->TryFindKernel(node, "");  //the last argument is ignored
    if (*kernel_create_info != nullptr) {
      return Status::OK();
    }
  }

  std::ostringstream errormsg;
  errormsg << "Failed to find kernel for " << node.OpType();
  if (node.Op() != nullptr)
    errormsg << "(" << node.Op()->since_version() << ")";
  if (!node.Name().empty()) errormsg << " (node " << node.Name() << ")";
  return Status(ONNXRUNTIME, FAIL, errormsg.str());
}

}  // namespace onnxruntime
