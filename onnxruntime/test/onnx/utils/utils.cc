// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "utils.h"
#include "strings_helper.h"
#include <core/platform/path_lib.h>

#include <cstdint>
#include <filesystem>

namespace onnxruntime {
namespace test {
namespace utils {
void ListEpDevices(const Ort::Env& env) {
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
    fprintf(stdout, "%s", device_info_msg.c_str());
  }
}

void RegisterExecutionProviderLibrary(Ort::Env& env,
                                      std::basic_string<ORTCHAR_T>& plugin_ep_names_and_libs,
                                      std::vector<std::string>& registered_plugin_eps) {
  if (!plugin_ep_names_and_libs.empty()) {
    std::unordered_map<std::string, std::string> ep_names_to_libs;
    ParseSessionConfigs(onnxruntime::ToUTF8String(plugin_ep_names_and_libs), ep_names_to_libs);
    if (ep_names_to_libs.size() > 0) {
      for (auto& pair : ep_names_to_libs) {
        const std::filesystem::path library_path = pair.second;
        const std::string registration_name = pair.first;
        Ort::Status status(Ort::GetApi().RegisterExecutionProviderLibrary(env, registration_name.c_str(), ToPathString(library_path.string()).c_str()));
        if (status.IsOK()) {
          registered_plugin_eps.push_back(registration_name);
        } else {
          fprintf(stderr, "Can't register %s plugin library: %s\n", registration_name.c_str(), status.GetErrorMessage().c_str());
        }
      }
    }
  }
}

void UnregisterExecutionProviderLibrary(Ort::Env& env, std::vector<std::string>& registered_plugin_eps) {
  for (auto& registration_name : registered_plugin_eps) {
    Ort::Status status(Ort::GetApi().UnregisterExecutionProviderLibrary(env, registration_name.c_str()));
    if (!status.IsOK()) {
      fprintf(stderr, "%s", status.GetErrorMessage().c_str());
    }
  }
}

}  // namespace utils
}  // namespace test
}  // namespace onnxruntime
