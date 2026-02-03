// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>

#include "../plugin_ep_utils.h"

class ExampleEpFactory;

/// <summary>
/// Example implementation of ONNX Mul. Does not handle many things like broadcasting.
/// </summary>
struct MulKernel {
  MulKernel(const OrtApi& ort_api, const OrtLogger& logger,
            const std::unordered_map<std::string, FloatInitializer>& float_initializers,
            std::string input0_name, std::string input1_name)
      : ort_api(ort_api),
        logger(logger),
        float_initializers(float_initializers),
        input0_name(input0_name),
        input1_name(input1_name) {}

  const FloatInitializer* TryGetSavedInitializer(const std::string& name) const;

  void GetInputDataAndShape(Ort::KernelContext kernel_context, size_t index,
                            /*out*/ gsl::span<const float>& data,
                            /*out*/ std::vector<int64_t>& shape) const;

  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api;
  const OrtLogger& logger;
  const std::unordered_map<std::string, FloatInitializer>& float_initializers;
  std::string input0_name;
  std::string input1_name;
};

/// <summary>
/// Kernel for EPContext nodes loaded from compiled models.
///
/// This example EP does not support EPContext inference - Compute() returns NOT_IMPLEMENTED.
/// A production EP would deserialize the ep_cache_context attribute and restore compiled state.
/// This kernel exists to clearly separate EPContext handling from MulKernel.
/// </summary>
struct EpContextKernel {
  EpContextKernel(const OrtApi& ort_api, const OrtLogger& logger)
      : ort_api(ort_api), logger(logger) {}

  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api;
  const OrtLogger& logger;
};

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

  std::unordered_map<std::string, std::unique_ptr<MulKernel>>& MulKernels() {
    return mul_kernels_;
  }

  std::unordered_map<std::string, std::unique_ptr<EpContextKernel>>& EpContextKernels() {
    return ep_context_kernels_;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                               _In_ const OrtMemoryDevice* memory_device,
                                                               _Outptr_ OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;

  static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(OrtEp* this_ptr,
                                                                        const OrtGraph* graph) noexcept;

  OrtStatus* CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                  /*out*/ gsl::span<OrtNode*> ep_context_nodes);

  OrtStatus* SaveConstantInitializers(const OrtGraph* graph);

  ExampleEpFactory& factory_;
  std::string name_;
  Config config_{};
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> mul_kernels_;
  std::unordered_map<std::string, std::unique_ptr<EpContextKernel>> ep_context_kernels_;
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
  std::string compatibility_info_;  // Cached compatibility string returned by GetCompiledModelCompatibilityInfo
};
