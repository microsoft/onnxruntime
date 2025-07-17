// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>

#include "example_plugin_ep_utils.h"

class ExampleEpFactory;
struct MulKernel;

/// <summary>
/// Example EP that can compile a single Mul operator.
/// </summary>
class ExampleEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context = false;
    // Other EP configs (typically extracted from OrtSessionOptions or OrtHardwareDevice(s))
  };

  ExampleEp(ExampleEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger);

  ~ExampleEp();

  std::unordered_map<std::string, std::unique_ptr<MulKernel>>& Kernels() {
    return kernels_;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                        _In_ const OrtMemoryInfo* memory_info,
                                        _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;

  static OrtStatus* CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                  _In_ const OrtMemoryDevice* memory_device,
                                                  _Outptr_ OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL ExampleEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
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

  OrtStatus* ExampleEp::SaveConstantInitializers(const OrtGraph* graph);

  ExampleEpFactory& factory_;
  std::string name_;
  Config config_{};
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> kernels_;
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
};
