// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph_transformer.h"

namespace onnxruntime {

// Information needed to construct MKL-DNN execution providers.
struct MKLDNNExecutionProviderInfo {
  bool create_arena{true};

  explicit MKLDNNExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  MKLDNNExecutionProviderInfo() = default;
};

// Logical device representation.
class MKLDNNExecutionProvider : public IExecutionProvider {
 public:
  explicit MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info);
  virtual ~MKLDNNExecutionProvider();

  std::string Type() const override {
    return onnxruntime::kMklDnnExecutionProvider;
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
