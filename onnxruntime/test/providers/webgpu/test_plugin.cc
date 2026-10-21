// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/webgpu/test_plugin.h"

#include <iostream>
#include <filesystem>
#include <string_view>
#include <mutex>
#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/common/path_string.h"
#include "core/platform/env.h"
#include "core/providers/providers.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_env.h"
#include "onnxruntime_cxx_api.h"
#include "core/session/utils.h"
#include "test/providers/webgpu/internal_test_api.h"
#include "test/unittest_util/test_dynamic_plugin_ep.h"
#include "test/util/include/test/test_environment.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime::test {

struct WebGpuTestPlugin::Impl {
  void* library = nullptr;
  BeginWebGpuTestContextFn begin_context = nullptr;
  EndWebGpuTestContextFn end_context = nullptr;
  ConfigureWebGpuTestEpFn configure = nullptr;
  InspectWebGpuTestEpFn inspect = nullptr;
  OrtEpFactory* ep_factory = nullptr;
  OrtEpDevice* device = nullptr;
  ReleaseEpApiFactoryFn release_factory = nullptr;
  std::unique_ptr<IExecutionProviderFactory> factory;

  ~Impl() {
    factory.reset();
    if (device) {
      Ort::GetApi().GetEpApi()->ReleaseEpDevice(device);
    }
    if (ep_factory) {
      Ort::Status status{release_factory(ep_factory)};
      if (!status.IsOK()) std::cerr << status.GetErrorMessage() << '\n';
    }
    if (library) {
      ORT_IGNORE_RETURN_VALUE(Env::Default().UnloadDynamicLibrary(library));
    }
  }
};

WebGpuTestPlugin::WebGpuTestPlugin()
    : impl_(std::make_unique<Impl>()) {
  const auto path = (std::filesystem::path{Env::Default().GetRuntimePath()} /
                     ToPathString(ORT_WEBGPU_INTERNAL_TEST_LIBRARY_PATH))
                        .native();
  ORT_THROW_IF_ERROR(Env::Default().LoadDynamicLibrary(path, false, &impl_->library));
  void* begin = nullptr;
  void* end = nullptr;
  ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(impl_->library, "BeginWebGpuTestContext", &begin));
  ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(impl_->library, "EndWebGpuTestContext", &end));
  impl_->end_context = reinterpret_cast<EndWebGpuTestContextFn>(end);
  impl_->begin_context = reinterpret_cast<BeginWebGpuTestContextFn>(begin);
  void* configure = nullptr;
  void* inspect = nullptr;
  ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(impl_->library, "ConfigureWebGpuTestEp", &configure));
  ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(impl_->library, "InspectWebGpuTestEp", &inspect));
  impl_->configure = reinterpret_cast<ConfigureWebGpuTestEpFn>(configure);
  impl_->inspect = reinterpret_cast<InspectWebGpuTestEpFn>(inspect);

  void* create = nullptr;
  void* release = nullptr;
  ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(impl_->library, "CreateEpFactories", &create));
  ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(impl_->library, "ReleaseEpFactory", &release));
  impl_->release_factory = reinterpret_cast<ReleaseEpApiFactoryFn>(release);
  size_t count = 0;
  Ort::ThrowOnError(reinterpret_cast<CreateEpApiFactoriesFn>(create)(
      "WebGpuInternalTestPlugin", OrtGetApiBase(), DefaultLoggingManager().DefaultLogger().ToExternal(),
      &impl_->ep_factory, 1, &count));
  ORT_ENFORCE(count == 1);

  // Use the selected plugin's hardware device without registering a second copy
  // of its shared allocators or data transfers in the process-wide environment.
  for (auto device : ort_env->GetEpDevices()) {
    if (std::string_view(device.EpName()) == "WebGpuExecutionProvider") {
      const OrtHardwareDevice* hardware[] = {Ort::GetApi().EpDevice_Device(device)};
      Ort::ThrowOnError(impl_->ep_factory->GetSupportedDevices(
          impl_->ep_factory, hardware, 1, &impl_->device, 1, &count));
      break;
    }
  }
  ORT_ENFORCE(impl_->device != nullptr, "The WebGPU test plugin did not expose a device.");
  const OrtEpDevice* devices[] = {impl_->device};
  ORT_THROW_IF_ERROR(CreateIExecutionProviderFactoryForEpDevices(
      static_cast<OrtEnv*>(*ort_env)->GetEnvironment(), devices, impl_->factory));
}

WebGpuTestPlugin::~WebGpuTestPlugin() = default;

std::unique_ptr<IExecutionProvider> WebGpuTestPlugin::MakeEp(
    const ConfigOptions& config, const WebGpuExecutionProviderTestOptions& test_options) const {
  void* context = nullptr;
  bool supports_dp4a = false;
  OrtSessionOptions options;
  const auto defaults = dynamic_plugin_ep_infra::GetDefaultEpOptions();
  InlinedVector<const char*> keys, values;
  for (const auto& [key, value] : defaults) {
    keys.push_back(key.c_str());
    values.push_back(value.c_str());
  }
  const OrtEpDevice* devices[] = {impl_->device};
  ORT_THROW_IF_ERROR(AddEpOptionsToSessionOptions(devices, keys, values, options.value));
  // The private helper takes session config keys, including the EP prefix.
  // Per-call settings override the infrastructure and device defaults.
  for (const auto& [key, value] : config.GetConfigOptionsMap()) {
    ORT_THROW_IF_ERROR(options.value.config_options.AddConfigEntry(key.c_str(), value.c_str()));
  }
  ORT_THROW_IF_ERROR(options.value.config_options.AddConfigEntry(webgpu::options::kPreserveDevice, "0"));
  Ort::ThrowOnError(impl_->begin_context(&options, test_options.max_storage_buffer_binding_size, &context, &supports_dp4a));
  auto release_context = gsl::finally([&]() { impl_->end_context(context); });
  auto ep = impl_->factory->CreateProvider(options, *DefaultLoggingManager().DefaultLogger().ToExternal());
  Ort::ThrowOnError(impl_->configure(const_cast<OrtEp*>(ep->GetOrtEp()),
                                     test_options.forced_matmul_algorithm ? static_cast<int>(*test_options.forced_matmul_algorithm) : -1));
  return ep;
}

WebGpuTestCapabilities WebGpuTestPlugin::Inspect(const IExecutionProvider& ep) const {
  WebGpuTestCapabilities result;
  Ort::ThrowOnError(impl_->inspect(ep.GetOrtEp(), &result));
  return result;
}

namespace {
std::unique_ptr<WebGpuTestPlugin> test_plugin;
std::mutex test_plugin_mutex;
class TestPluginEnvironment : public testing::Environment {
  void TearDown() override { test_plugin.reset(); }
};
[[maybe_unused]] auto* const test_plugin_environment = testing::AddGlobalTestEnvironment(new TestPluginEnvironment);
}  // namespace

WebGpuTestPlugin& GetWebGpuTestPlugin() {
  std::lock_guard lock(test_plugin_mutex);
  if (!test_plugin) test_plugin = std::make_unique<WebGpuTestPlugin>();
  return *test_plugin;
}

}  // namespace onnxruntime::test
