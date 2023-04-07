// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers_fwd.h"
#include "core/common/optional.h"
#include "core/framework/execution_provider.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_api_helper.h"
#include "core/providers/nnapi/nnapi_provider_factory.h"

struct NnApi;
namespace onnxruntime {
namespace nnapi {
class Model;
}

class NnapiExecutionProvider : public IExecutionProvider {
 public:
  explicit NnapiExecutionProvider(uint32_t nnapi_flags,
                                  const optional<std::string>& partitioning_stop_ops_list = {});

  virtual ~NnapiExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const IKernelLookup& /*kernel_lookup*/) const override;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
#endif

  uint32_t GetNNAPIFlags() const { return nnapi_flags_; }

  DataLayout GetPreferredLayout() const override;

 private:
  // The bit flags which define bool options for NNAPI EP, bits are defined as
  // NNAPIFlags in include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h
  const uint32_t nnapi_flags_;

  const std::unordered_set<std::string> partitioning_stop_ops_;

  std::unordered_map<std::string, std::unique_ptr<onnxruntime::nnapi::Model>> nnapi_models_;

  // For Android NNAPI and stub implementation.
  const NnApi* nnapi_handle_ = nullptr;
  nnapi::DeviceWrapperVector nnapi_target_devices_;
  nnapi::TargetDeviceOption target_device_option_;
};
}  // namespace onnxruntime
