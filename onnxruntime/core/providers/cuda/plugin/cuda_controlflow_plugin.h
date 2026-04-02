// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plugin EP control flow kernel wrappers for If, Loop, and Scan.
// These delegate to OrtEpApi::CreateIfKernel/CreateLoopKernel/CreateScanKernel
// instead of inheriting from CPU base classes.

#pragma once

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace cuda {
namespace plugin {

// ===================================================================
// If kernel wrapper — delegates to OrtEpApi::CreateIfKernel
// ===================================================================

class PluginIfKernel : public OpKernel {
 public:
  explicit PluginIfKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status CreateControlFlowKernelImpl(const OrtKernelInfo* info, OrtKernelImpl** impl) override;

  Status Compute(OpKernelContext*) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Plugin If kernel should not be called directly");
  }
};

// ===================================================================
// Loop kernel helper — provides GPU ConcatOutput via cudaMemcpyAsync
// ===================================================================

class PluginLoopHelper : public OrtLoopKernelHelper {
 public:
  PluginLoopHelper();

  static void ORT_API_CALL ReleaseImpl(_In_ OrtLoopKernelHelper* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL ConcatOutputImpl(
      _In_ OrtLoopKernelHelper* this_ptr,
      _In_opt_ void* stream_handle,
      _In_reads_(num_per_iteration_outputs) const OrtValue* const* per_iteration_outputs,
      _In_ size_t num_per_iteration_outputs,
      _Out_writes_bytes_all_(output_size_in_bytes) void* output,
      _In_ size_t output_size_in_bytes) noexcept;
};

class PluginLoopKernel : public OpKernel {
 public:
  explicit PluginLoopKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status CreateControlFlowKernelImpl(const OrtKernelInfo* info, OrtKernelImpl** impl) override;

  Status Compute(OpKernelContext*) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Plugin Loop kernel should not be called directly");
  }
};

// ===================================================================
// Scan kernel helper — provides GPU Transpose via CUDA kernel
// ===================================================================

class PluginScanHelper : public OrtScanKernelHelper {
 public:
  PluginScanHelper();

  static void ORT_API_CALL ReleaseImpl(_In_ OrtScanKernelHelper* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL TransposeImpl(
      _In_ OrtScanKernelHelper* this_ptr,
      _In_reads_(num_permutation_elems) const size_t* permutation,
      _In_ size_t num_permutation_elems,
      _In_ const OrtValue* input,
      _In_opt_ OrtSyncStream* stream,
      _Inout_ OrtValue* output) noexcept;
};

class PluginScanKernel : public OpKernel {
 public:
  explicit PluginScanKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status CreateControlFlowKernelImpl(const OrtKernelInfo* info, OrtKernelImpl** impl) override;

  Status Compute(OpKernelContext*) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Plugin Scan kernel should not be called directly");
  }
};

// GPU transpose helper (defined in cuda_controlflow_plugin.cu)
OrtStatus* LaunchTransposeKernel(const void* input, void* output,
                                 const int64_t* input_shape, const size_t* permutation,
                                 size_t num_dims, size_t element_size, size_t total_elements,
                                 cudaStream_t stream);

}  // namespace plugin
}  // namespace cuda
}  // namespace onnxruntime
