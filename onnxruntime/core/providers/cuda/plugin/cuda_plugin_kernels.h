// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "cuda_plugin_utils.h"

namespace onnxruntime {
namespace cuda_plugin {

using PluginKernelCreateFn = OrtStatus*(ORT_API_CALL*)(void*, const OrtKernelInfo*, OrtKernelImpl**) noexcept;

/// Resolve a plugin kernel creation function by operator type/domain.
/// Returns nullptr if the operator is not implemented by the CUDA plugin.
PluginKernelCreateFn ResolvePluginKernelCreateFn(std::string_view op_type, std::string_view domain = "");

/// Create the CUDA kernel registry via the adapter-backed registration path.
OrtStatus* CreateCudaKernelRegistryFromOrtTables(const OrtEpApi& ep_api,
                                                 const char* ep_name,
                                                 void* create_kernel_state,
                                                 OrtKernelRegistry** out_registry);

/// Dispatcher entrypoint used by the EP factory.
OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* create_kernel_state,
                                    OrtKernelRegistry** out_registry);

}  // namespace cuda_plugin
}  // namespace onnxruntime
