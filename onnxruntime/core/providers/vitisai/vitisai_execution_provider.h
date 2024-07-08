// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard headers/libs.
#include <ctime>
#include <vector>
#include <memory>
#include <set>
#include <string>

// 1st-party headers/libs.
// #include "core/framework/session_options.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/common/inlined_containers_fwd.h"

// we cannot include vaip/vaip.hpp here because header file referred by
// onnxruntime_pybind_state_common.cc
namespace vaip_core {
template <typename T>
class DllSafe;
class ExecutionProvider;
}  // namespace vaip_core
namespace onnxruntime {
// Logical device representation.
class VitisAIExecutionProvider : public IExecutionProvider {
 public:
  explicit VitisAIExecutionProvider(const ProviderOptions& info);
  // explicit VitisAIExecutionProvider(const ProviderOptions& info,
  //     const SessionOptions* p_sess_opts = nullptr);
  ~VitisAIExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                                const IKernelLookup& /*kernel_lookup*/) const override;

  int GetDeviceId() const { return 0; }

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  // This method is called after both `GetComputeCapabilityOps()` and `Compile()`.
  // This timing is required to work with both compliation-based EPs and non-compilation-based EPs.
  const InlinedVector<const Node*> GetEpContextNodes() const override;

 private:
  void CreateKernelRegistry();
  using my_ep_t = vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>;
  using my_ep_uptr_t = std::shared_ptr<my_ep_t>;
  // we have to hide the implementation by forward declaration.
  mutable my_ep_uptr_t execution_providers_;
  mutable ProviderOptions info_;
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
  std::shared_ptr<KernelRegistry> registry_;
  std::set<std::string> vitisai_optypes_;
  // EP context related.
  bool ep_ctx_enabled_ = false;
  bool ep_ctx_embed_mode_ = true;
  std::string ep_ctx_model_path_cfg_{""};
  mutable std::string backend_cache_data_{""};
  mutable PathString model_path_str_{};
  mutable PathString ep_ctx_model_file_loc_{};
  mutable std::unique_ptr<onnxruntime::Model> p_ep_ctx_model_;
  mutable std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_ep_ctx_model_proto_;
  // It might need to be called before loading
  // the EP context model that is compiled AOT/offline.
  void LoadEPContexModelFromFile() const;
  void PrepareEPContextEnablement(const onnxruntime::GraphViewer&) const;
  void FulfillEPContextEnablement(const std::vector<FusedNodeAndGraph>&);
};

}  // namespace onnxruntime
