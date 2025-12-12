// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>
#include <memory>
#include <string>
#include <unordered_map>

#include "api.h"

namespace onnxruntime {
struct IExecutionProvider;
}

namespace onnxruntime {
namespace webgpu {
namespace ep {

class Factory;

/// <summary>
/// A bridge class between the EP API and the WebGPU EP implementation.
/// </summary>
class Ep : public OrtEp {
 public:
  struct Config {
    // Add per-kernel EP specific configuration options here
    // For example:
    // bool enable_profiling = false;
    // int max_batch_size = 1;
  };

  // Do not use a std::unique_ptr for impl_ because this requires the actual type definition.
  Ep(IExecutionProvider* impl, Factory& factory, const OrtLogger& logger, const Config& config);

  ~Ep();

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

  IExecutionProvider* impl_;
  Factory& factory_;
  const OrtLogger& logger_;
  Config config_{};
};

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
