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
    const std::filesystem::path library_path =
#if _WIN32
        "example_plugin_ep.dll";
#else
        "libexample_plugin_ep.so";
#endif
    const std::string registration_name = "example_ep";
  };

  static ExamplePluginInfo example_ep_info;

  // get the OrtEpDevice for an arbitrary EP from the environment
  static void GetEp(Ort::Env& env, const std::string& ep_name, const OrtEpDevice*& ep_device);

  // Register the example EP library, get the OrtEpDevice for it, and return a unique pointer that will
  // automatically unregister the EP library.
  static void RegisterAndGetExampleEp(Ort::Env& env, RegisteredEpDeviceUniquePtr& example_ep);
};
}  // namespace test
}  // namespace onnxruntime
