// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "core/framework/execution_provider.h"

namespace onnxruntime {

class SNPEExecutionProvider : public IExecutionProvider {
 public:
  explicit SNPEExecutionProvider(const ProviderOptions& provider_options_map);
  virtual ~SNPEExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const IKernelLookup& kernel_lookup) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unordered_map<std::string, std::string> GetRuntimeOptions() const { return runtime_options_; }

 private:
  ProviderOptions runtime_options_;
};
}  // namespace onnxruntime
