// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

class EpFactoryVirtualGpu;
struct VirtualCompiledAdd;

/// <summary>
/// Example EP for a virtual GPU OrtHardwareDevice that was created by the EP factory itself (not ORT).
/// Can only compile/execute a single Add node. Only used to test that an EP can provide additional hardware devices.
/// </summary>
class EpVirtualGpu : public OrtEp {
 public:
  struct Config {
    bool enable_ep_context = false;
    // Other EP configs (typically extracted from OrtSessionOptions or OrtHardwareDevice(s))
  };

  EpVirtualGpu(EpFactoryVirtualGpu& factory, const Config& config, const OrtLogger& logger);
  ~EpVirtualGpu();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }

  std::unordered_map<std::string, std::unique_ptr<VirtualCompiledAdd>>& GetCompiledSubgraphs() {
    return compiled_subgraphs_;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;

  OrtStatus* CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                  /*out*/ gsl::span<OrtNode*> ep_context_nodes);

  Config config_{};
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtModelEditorApi& model_editor_api_;
  std::string name_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<VirtualCompiledAdd>> compiled_subgraphs_;
};
