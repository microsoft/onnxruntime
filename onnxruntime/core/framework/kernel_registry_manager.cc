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
  auto create_error_message = [&node](const std::string& error) {
    std::ostringstream errormsg;
    errormsg << error << node.OpType();
    if (node.Op() != nullptr) errormsg << "(" << node.Op()->since_version() << ")";
    if (!node.Name().empty()) errormsg << " (node " << node.Name() << ")";
    return errormsg.str();
  };

  const std::string& ptype = node.GetExecutionProviderType();
  if (ptype.empty()) {
    return Status(ONNXRUNTIME, FAIL,
                  create_error_message("The node is not placed on any Execution Provider, "
                                       "therefore, can't find a suitable kernel for "));
  }

  Status status;
  {
    for (auto& registry : custom_kernel_registries_) {
      status = registry->TryCreateKernel(node, execution_provider, session_state.GetConstantInitializedTensors(),
                                         session_state.GetOrtValueNameIdxMap(), session_state.GetFuncMgr(), session_state.GetDataTransferMgr(), op_kernel);
      if (status.IsOK()) {
        return status;
      }
    }
  }

  KernelRegistry* p = nullptr;
  auto iter = provider_type_to_registry_.find(ptype);
  if (iter != provider_type_to_registry_.end()) p = iter->second.get();
  if (p != nullptr) {
    status = p->TryCreateKernel(node, execution_provider, session_state.GetConstantInitializedTensors(),
                                session_state.GetOrtValueNameIdxMap(), session_state.GetFuncMgr(), session_state.GetDataTransferMgr(), op_kernel);
    if (status.IsOK()) {
      return status;
    }
  }

  return Status(ONNXRUNTIME, NOT_IMPLEMENTED, create_error_message("Failed to find kernel for "));
}

Status KernelRegistryManager::RegisterKernels(const ExecutionProviders& execution_providers) {
  for (auto& provider : execution_providers) {
    auto iter = provider_type_to_registry_.find(provider->Type());
    if (iter != provider_type_to_registry_.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "found duplicated provider ", provider->Type(),
                             " in KernelRegistryManager");
    }

    auto registry = provider->GetKernelRegistry();
    if (!registry) {
      continue;
    }

    provider_type_to_registry_.insert(std::make_pair(provider->Type(), registry));
  }
  return Status::OK();
}

void KernelRegistryManager::RegisterKernelRegistry(std::shared_ptr<KernelRegistry> kernel_registry) {
  if (nullptr == kernel_registry) {
    return;
  }
  custom_kernel_registries_.push_front(kernel_registry);
}

bool KernelRegistryManager::HasImplementationOf(const KernelRegistryManager& r, const Node& node, const std::string& provider_type) {
  std::vector<const KernelRegistry*> kernel_registries = r.GetKernelRegistriesByProviderType(provider_type);
  return std::any_of(kernel_registries.begin(), kernel_registries.end(), [&](const KernelRegistry* kernel_registry) {
    return KernelRegistry::HasImplementationOf(*kernel_registry, node, provider_type);
  });
}

Status KernelRegistryManager::SearchKernelRegistry(const onnxruntime::Node& node,
                                                   /*out*/ const KernelCreateInfo** kernel_create_info) const {
  const std::string& ptype = node.GetExecutionProviderType();
  if (ptype.empty()) {
    return Status(ONNXRUNTIME, FAIL, "The node is not placed on any Execution Provider");
  }
  Status status;
  {
    for (auto& registry : custom_kernel_registries_) {
      status = registry->TryFindKernel(node, std::string(), kernel_create_info);
      if (status.IsOK()) return status;
    }
  }

  KernelRegistry* p = nullptr;
  auto iter = provider_type_to_registry_.find(ptype);
  if (iter != provider_type_to_registry_.end()) p = iter->second.get();
  if (p != nullptr) {
    status = p->TryFindKernel(node, std::string(), kernel_create_info);
    if (status.IsOK()) return status;
  }

  std::ostringstream errormsg;
  errormsg << "Failed to find kernel for " << node.OpType();
  if (node.Op() != nullptr) errormsg << "(" << node.Op()->since_version() << ")";
  if (!node.Name().empty()) errormsg << " (node " << node.Name() << ").";
  if (!status.IsOK()) errormsg << status.ErrorMessage();
  return Status(ONNXRUNTIME, NOT_IMPLEMENTED, errormsg.str());
}

}  // namespace onnxruntime
