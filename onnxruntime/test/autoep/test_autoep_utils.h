// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <string>

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace test {

using RegisteredEpDeviceUniquePtr = std::unique_ptr<const OrtEpDevice, std::function<void(const OrtEpDevice*)>>;

struct Utils {
  struct ExamplePluginInfo {
    ExamplePluginInfo(std::filesystem::path lib_path, const char* reg_name, const char* ep_name);

    std::filesystem::path library_path;
    std::string registration_name;
    std::string ep_name;
  };

  static const ExamplePluginInfo example_ep_info;                  // example_plugin_ep.dll
  static const ExamplePluginInfo example_ep_virt_gpu_info;         // example_plugin_ep_virt_gpu.dll
  static const ExamplePluginInfo example_ep_kernel_registry_info;  // example_plugin_ep_kernel_registry.dll

  // get the OrtEpDevice for an arbitrary EP from the environment
  static void GetEp(Ort::Env& env, const std::string& ep_name, const OrtEpDevice*& ep_device);

  // Register the example EP library, get the OrtEpDevice for it, and return a unique pointer that will
  // automatically unregister the EP library.
  static void RegisterAndGetExampleEp(Ort::Env& env, const ExamplePluginInfo& ep_info,
                                      RegisteredEpDeviceUniquePtr& example_ep);

  struct ExampleEpHooks {
    using ResetSyncCountFn = void (*)();
    using GetSyncCountFn = uint64_t (*)();

    ResetSyncCountFn reset_sync_count{};
    GetSyncCountFn get_sync_count{};
  };

  using LoadExampleEpHooksPtr = std::unique_ptr<ExampleEpHooks, std::function<void(ExampleEpHooks*)>>;

  static void LoadExampleEpHooks(const Utils::ExamplePluginInfo& ep_info,
                                 LoadExampleEpHooksPtr& example_ep_hooks);
};

}  // namespace test
}  // namespace onnxruntime
