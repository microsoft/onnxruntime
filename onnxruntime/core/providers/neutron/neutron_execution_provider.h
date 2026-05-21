// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/providers/neutron/neutron_provider_factory.h"

#if NEUTRON_AARCH64
#include "neutron/NeutronDriver.h"
#endif

namespace onnxruntime {

class NeutronExecutionProvider : public IExecutionProvider {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NeutronExecutionProvider);

  explicit NeutronExecutionProvider(NeutronProviderOptions neutron_options);
  virtual ~NeutronExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* /* resource_accountant */) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  bool IsOfflinePacked() const {
    return neutron_options_.offline_packed;
  }

 private:
  NeutronProviderOptions neutron_options_;
  enum class NEUTRON_STATE { FAILED,
                             OP_ONLY,
                             OK };
  NEUTRON_STATE neutron_state_{NEUTRON_STATE::FAILED};
  uint32_t node_number_{0};
};

Status RegisterNeutronKernels(KernelRegistry& kernel_registry);

}  // namespace onnxruntime
