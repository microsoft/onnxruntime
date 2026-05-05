// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"

namespace onnxruntime {
namespace cuda_plugin {

/// Create the CUDA kernel registry using self-registered BuildKernelCreateInfo<>
/// entries from PluginKernelCollector. All compiled kernel .cc files automatically
/// contribute their registrations via the macro overrides in cuda_kernel_adapter.h.
OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* create_kernel_state,
                                    OrtKernelRegistry** out_registry);

}  // namespace cuda_plugin
}  // namespace onnxruntime
