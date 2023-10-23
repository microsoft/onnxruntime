// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"

struct pthreadpool;
namespace onnxruntime {

namespace js {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

}  // namespace js

struct JsExecutionProviderInfo {
  JsExecutionProviderInfo(const ProviderOptions& po) {
    auto it = po.find("preferred_layout");
    if (it != po.end()) {
      auto& value = it->second;
      if (value == "NCHW") {
        data_layout = DataLayout::NCHW;
      } else if (value == "NHWC") {
        data_layout = DataLayout::NHWC;
      }
    }
  }

  // JSEP default preferred layout is NHWC
  DataLayout data_layout = DataLayout::NHWC;
};

class JsExecutionProvider : public IExecutionProvider {
 public:
  JsExecutionProvider(const JsExecutionProviderInfo& info);
  ~JsExecutionProvider() override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& /*kernel_lookup*/) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  DataLayout GetPreferredLayout() const override { return preferred_data_layout_; }

  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  // JSEP disallow concurrent run because actual implementation (eg. WebGPU backend) relies on global states to work,
  // and concurrent run with async function may mess up the states and cause undefined behavior.
  bool ConcurrentRunSupported() const override { return false; }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
  DataLayout preferred_data_layout_;
};

}  // namespace onnxruntime
