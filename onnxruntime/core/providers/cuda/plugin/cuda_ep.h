// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"
#include "cuda_graph_plugin.h"

#include <atomic>
#include <string>
#include <unordered_map>
#include <mutex>

namespace onnxruntime {
namespace cuda_plugin {

class CudaEpFactory;

/// CUDA execution provider implementation using public OrtEp interface.
class CudaEp : public OrtEp {
 public:
  /// Configuration parameters for the CUDA EP, parsed from session options.
  struct Config {
    bool prefer_nhwc = false;                         ///< Use NHWC data layout when available.
    bool use_tf32 = true;                             ///< Enable TF32 math on Ampere+ GPUs.
    bool enable_skip_layer_norm_strict_mode = false;  ///< Strict mode for SkipLayerNorm kernel.
    int device_id = 0;                                ///< CUDA device ordinal.
    int cudnn_conv_algo = 0;                          ///< cuDNN convolution algorithm selection.
    bool cudnn_conv1d_pad_to_nc1d = false;            ///< Pad 1D convolutions to NC1D format.
    bool enable_cuda_graph = false;                   ///< Enable CUDA graph capture/replay.
    int min_num_runs_before_cuda_graph_capture = 1;   ///< Warm-up runs before graph capture.
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

  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(
      OrtEp* this_ptr, const char* domain, const char* op_type,
      OrtEpDataLayout target_data_layout, int* should_convert) noexcept;

  static OrtStatus* ORT_API_CALL OnRunEndImpl(
      OrtEp* this_ptr, const ::OrtRunOptions* run_options, bool sync_stream) noexcept;

  // CUDA Graph helpers
  CudaGraphAnnotation_t GetAnnotationId(const ::OrtRunOptions* run_options) const;
  bool IsGraphCaptureAllowed(CudaGraphAnnotation_t annotation_id) const;

  CudaEpFactory& factory_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;

  // CUDA Graph state
  std::atomic<bool> cuda_graph_enabled_{false};
  int min_runs_before_capture_ = 1;
  CUDAGraphManager cuda_graph_manager_;
  std::unordered_map<CudaGraphAnnotation_t, int> graph_id_to_run_count_;
  bool is_capturing_ = false;
  CudaGraphAnnotation_t capturing_annotation_id_ = kCudaGraphAnnotationDefault;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
