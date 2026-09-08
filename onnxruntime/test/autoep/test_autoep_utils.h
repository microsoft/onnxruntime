// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

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

// Runs the mul_1.onnx inference test with ep_name selected in two ways:
// - automatically through "test.ep_to_select", when test_auto_select is true.
// - explicitly through SessionOptionsAppendExecutionProvider_V2.
// Both paths pass provider_options to the EP: as prefixed session options on the auto-selection path and as
// ep_options on the V2 path. The V2 path uses the devices chosen by select_devices, or the first OrtEpDevice
// advertised for ep_name when no selector is provided. If library_path is provided, the helper registers the
// plugin EP library before creating the session.
// v2_session_checker runs after the V2 session is created and before inference.
// disable_cpu_ep_fallback applies to both paths. It requires the selected EP to handle the entire graph, preventing
// a false pass caused by unsupported nodes falling back to the ORT CPU EP.
void RunBasicTest(const std::string& ep_name, std::optional<std::filesystem::path> library_path,
                  const Ort::KeyValuePairs& provider_options = Ort::KeyValuePairs{},
                  const std::function<void(std::vector<const OrtEpDevice*>&)>& select_devices = nullptr,
                  bool test_auto_select = true,
                  const std::function<void(Ort::Session&)>& v2_session_checker = nullptr,
                  bool disable_cpu_ep_fallback = false);

}  // namespace test
}  // namespace onnxruntime
