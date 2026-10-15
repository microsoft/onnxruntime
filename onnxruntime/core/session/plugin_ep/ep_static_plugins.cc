// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_static_plugins.h"

#include "core/graph/constants.h"
#include "core/session/plugin_ep/ep_library_static_plugin.h"

// Entry points of the plugin execution providers that are statically linked into this build.
//
// A statically linked provider compiles its CreateEpFactories and ReleaseEpFactory entry points with a provider
// specific `WebGpu_` style prefix (see ORT_PLUGIN_EP_STATICALLY_LINKED in the provider's api.cc) so that several
// providers can be linked into one binary without their entry points colliding. Declare them here rather than
// including a provider header, so that ORT core does not take a compile time dependency on provider internals.
#if defined(USE_WEBGPU) && defined(ORT_WEBGPU_STATIC_PLUGIN)
extern "C" {
OrtStatus* WebGpu_CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                    const OrtLogger* default_logger, OrtEpFactory** factories,
                                    size_t max_factories, size_t* num_factories) noexcept;
OrtStatus* WebGpu_ReleaseEpFactory(OrtEpFactory* factory) noexcept;
}
#endif

namespace onnxruntime {
std::vector<std::unique_ptr<EpLibrary>> CreateStaticPluginEpLibraries() {
  std::vector<std::unique_ptr<EpLibrary>> ep_libraries;

#if defined(USE_WEBGPU) && defined(ORT_WEBGPU_STATIC_PLUGIN)
  // Register under the EP name, matching what an internal EP does. ORT core picks the registration name because
  // there is no application call site to supply one.
  ep_libraries.push_back(std::make_unique<EpLibraryStaticPlugin>(kWebGpuExecutionProvider,
                                                                 &WebGpu_CreateEpFactories,
                                                                 &WebGpu_ReleaseEpFactory));
#endif

  return ep_libraries;
}
}  // namespace onnxruntime
