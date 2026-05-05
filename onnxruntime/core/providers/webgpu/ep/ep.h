// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

class Factory;

/// <summary>
/// A bridge class between the EP API and the WebGPU EP implementation.
/// </summary>
class Ep : public onnxruntime::ep::adapter::Ep {
 public:
  struct Config {
    AllocatorPtr cpu_allocator;
    AllocatorPtr device_allocator;
    AllocatorPtr initializer_allocator;
  };

  Ep(std::unique_ptr<IExecutionProvider> impl, Factory& factory, const OrtLogger& logger, const Config& config);

  inline const OrtLogger& GetOrtLogger() const noexcept {
    return logger_;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      _In_ OrtEp* this_ptr,
      _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                            _Out_ OrtEpDataLayout* preferred_data_layout) noexcept;

  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                                  _In_z_ const char* domain,
                                                                  _In_z_ const char* op_type,
                                                                  _In_ OrtEpDataLayout target_data_layout,
                                                                  _Outptr_ int* should_convert) noexcept;

  static OrtStatus* ORT_API_CALL OnRunStartImpl(_In_ OrtEp* this_ptr,
                                                _In_ const OrtRunOptions* run_options) noexcept;

  static OrtStatus* ORT_API_CALL OnRunEndImpl(_In_ OrtEp* this_ptr,
                                              _In_ const OrtRunOptions* run_options,
                                              _In_ bool sync_stream) noexcept;

  static OrtStatus* ORT_API_CALL IsConcurrentRunSupportedImpl(_In_ OrtEp* this_ptr,
                                                              _Out_ bool* is_concurrent_run_supported) noexcept;

  static bool ORT_API_CALL IsGraphCaptureEnabledImpl(_In_ const OrtEp* this_ptr) noexcept;

  static bool ORT_API_CALL IsGraphCapturedImpl(_In_ const OrtEp* this_ptr,
                                               _In_ int graph_annotation_id) noexcept;

  static OrtStatus* ORT_API_CALL ReplayGraphImpl(_In_ OrtEp* this_ptr,
                                                 _In_ int graph_annotation_id) noexcept;

  static OrtGraphCaptureNodeAssignmentPolicy ORT_API_CALL GetGraphCaptureNodeAssignmentPolicyImpl(
      _In_ const OrtEp* this_ptr) noexcept;

  Factory& factory_;
  const OrtLogger& logger_;
  Config config_{};
};

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
