// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"

namespace onnxruntime {
namespace cuda_plugin {

/// Create the CUDA kernel registry with initial validation kernels.
/// For Stage 1, this registers Relu and Add kernels.
OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* create_kernel_state,
                                    OrtKernelRegistry** out_registry);

}  // namespace cuda_plugin
}  // namespace onnxruntime
