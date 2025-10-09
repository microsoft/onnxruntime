// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

class EpFactoryVirtualGpu;
struct AddImpl;

/// <summary>
/// Example EP for a virtual GPU OrtHardwareDevice that was created by the EP factory itself (not ORT).
/// Does not currently execute any nodes. Only used to test that an EP can provide ORT additional hardware devices.
/// </summary>
class EpVirtualGpu : public OrtEp {
 public:
  EpVirtualGpu(EpFactoryVirtualGpu& factory, const OrtLogger& logger);
  ~EpVirtualGpu();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }

  std::unordered_map<std::string, std::unique_ptr<AddImpl>>& GetCompiledSubgraphs() {
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

  EpFactoryVirtualGpu& factory_;
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  std::string name_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<AddImpl>> compiled_subgraphs_;
};
