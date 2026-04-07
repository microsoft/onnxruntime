// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"
#include "ep/adapters.h"

#include <string>
#include <mutex>

namespace onnxruntime {
namespace cuda_plugin {

class CudaEpFactory;

/// CUDA execution provider implementation using public OrtEp interface.
class CudaEp : public onnxruntime::ep::adapter::Ep {
 public:
  /// Configuration parameters for the CUDA EP, parsed from session options.
  struct Config {
    bool prefer_nhwc = false;                         ///< Use NHWC data layout when available.
    bool use_tf32 = true;                             ///< Enable TF32 math on Ampere+ GPUs.
    bool enable_skip_layer_norm_strict_mode = false;  ///< Strict mode for SkipLayerNorm kernel.
    int device_id = 0;                                ///< CUDA device ordinal.
    int cudnn_conv_algo = 0;                          ///< cuDNN convolution algorithm selection.
    bool cudnn_conv_use_max_workspace = true;         ///< Use maximum workspace for cuDNN conv algo search.
    bool cudnn_conv1d_pad_to_nc1d = false;            ///< Pad 1D convolutions to NC1D format.
    bool fuse_conv_bias = false;                      ///< Enable cuDNN frontend conv+bias fusion.
    int sdpa_kernel = 0;                              ///< Attention backend bitmask override.
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

  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(
      OrtEp* this_ptr, const char* domain, const char* op_type,
      OrtEpDataLayout target_data_layout, int* should_convert) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEp* this_ptr, const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL SyncImpl(OrtEp* this_ptr) noexcept;

  CudaEpFactory& factory_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
