// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <string>

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace test {

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

  static void GetEp(Ort::Env& env, const std::string& ep_name, const OrtEpDevice*& ep_device);
  static void RegisterAndGetExampleEp(Ort::Env& env, const OrtEpDevice*& example_ep);
};
}  // namespace test
}  // namespace onnxruntime
