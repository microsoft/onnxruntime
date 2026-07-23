// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file provides the CreateCudaKernelRegistry entrypoint for the CUDA plugin EP.
//
// Kernel registration is now fully automatic: each compiled kernel .cc file's
// ONNX_OPERATOR_*_KERNEL_EX macro expansion creates a BuildKernelCreateInfo<>()
// template specialization and auto-registers it in PluginKernelCollector via
// the macro overrides in cuda_kernel_adapter.h.
//
// CreateCudaKernelRegistry iterates the collector and registers each entry
// into an adapter::KernelRegistry which is then returned to the EP factory.

#include "cuda_plugin_kernels.h"
#include "cuda_stream_plugin.h"
#include "cuda_kernel_adapter.h"

// Define the BuildKernelCreateInfo<void>() sentinel in onnxruntime::cuda.
// This is normally defined in cuda_execution_provider.cc (excluded from plugin).
// The NHWC registration tables reference it as a placeholder to prevent empty arrays.
namespace onnxruntime::cuda {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}
}  // namespace onnxruntime::cuda

namespace onnxruntime {
namespace cuda_plugin {

OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& /*ep_api*/,
                                    const char* /*ep_name*/,
                                    void* /*create_kernel_state*/,
                                    OrtKernelRegistry** out_registry) {
  *out_registry = nullptr;

  EXCEPTION_TO_STATUS_BEGIN

  // adapter::KernelRegistry wraps OrtKernelRegistry via the Ort C++ API.
  ::onnxruntime::ep::adapter::KernelRegistry registry;

  // Iterate all self-registered BuildKernelCreateInfoFn pointers.
  auto entries = ::onnxruntime::cuda::PluginKernelCollector::Instance().Entries();
  for (auto build_fn : entries) {
    ::onnxruntime::ep::adapter::KernelCreateInfo info = build_fn();
    if (info.kernel_def != nullptr) {  // filter the BuildKernelCreateInfo<void> sentinel
      ORT_THROW_IF_ERROR(registry.Register(std::move(info)));
    }
  }

  *out_registry = registry.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
