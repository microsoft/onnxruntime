// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"

#include <string>

namespace onnxruntime {
namespace cuda_plugin {

class CudaEpFactory;

/// CUDA execution provider implementation using public OrtEp interface.
class CudaEp : public OrtEp {
 public:
  struct Config {
    bool prefer_nhwc = false;
    bool use_tf32 = true;
    bool enable_skip_layer_norm_strict_mode = false;
    int device_id = 0;
  };

  CudaEp(CudaEpFactory& factory, const Config& config, const OrtLogger& logger);
  ~CudaEp();

  const char* GetEpName() const { return name_.c_str(); }
  const Config& GetConfig() const { return config_; }

 private:
  // OrtEp callback implementations
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* this_ptr, const OrtGraph* graph,
      OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      OrtEp* this_ptr,
      const OrtKernelRegistry** kernel_registry) noexcept;

  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(
      OrtEp* this_ptr, OrtEpDataLayout* preferred_data_layout) noexcept;

  static OrtStatus* ORT_API_CALL OnRunStartImpl(
      OrtEp* this_ptr, const ::OrtRunOptions* run_options) noexcept;

  static OrtStatus* ORT_API_CALL OnRunEndImpl(
      OrtEp* this_ptr, const ::OrtRunOptions* run_options, bool sync_stream) noexcept;

  CudaEpFactory& factory_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
