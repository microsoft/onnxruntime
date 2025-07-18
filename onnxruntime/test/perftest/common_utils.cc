// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/perftest/utils.h"
#include "test/perftest/strings_helper.h"
#include <core/platform/path_lib.h>

#include <cstdint>

#include <filesystem>

namespace onnxruntime {
namespace perftest {
namespace utils {

void list_devices(Ort::Env& env) {
  std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

  for (size_t i = 0; i < ep_devices.size(); ++i) {
    auto device = ep_devices[i];
    std::string device_info_msg = "===== device id " + std::to_string(i) + " ======\n";
    device_info_msg += "name: " + std::string(device.EpName()) + "\n";
    device_info_msg += "vendor: " + std::string(device.EpVendor()) + "\n";

    auto metadata = device.EpMetadata();
    std::unordered_map<std::string, std::string> metadata_entries = metadata.GetKeyValuePairs();
    if (!metadata_entries.empty()) {
      device_info_msg += "metadata:\n";
    }

    for (auto& entry : metadata_entries) {
      device_info_msg += "  " + entry.first + ": " + entry.second + "\n";
    }
    device_info_msg += "\n";
    fprintf(stdout, device_info_msg.c_str());
  }
}

bool RegisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config) {
  if (!test_config.plugin_ep_names_and_libs.empty()) {
    std::unordered_map<std::string, std::string> ep_names_to_libs;
    ParseSessionConfigs(ToUTF8String(test_config.plugin_ep_names_and_libs), ep_names_to_libs);
    if (ep_names_to_libs.size() > 0) {
      for (auto& pair : ep_names_to_libs) {
        const std::filesystem::path library_path = pair.second;
        const std::string registration_name = pair.first;
        env.RegisterExecutionProviderLibrary(registration_name.c_str(), Utf8ToWide(library_path.string()));
        test_config.registered_plugin_eps.push_back(registration_name);
      }
    }
  }
  return true;
}

bool UnregisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config) {
  for (auto& registration_name : test_config.registered_plugin_eps) {
    env.UnregisterExecutionProviderLibrary(registration_name.c_str());
  }
  return true;
}

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
