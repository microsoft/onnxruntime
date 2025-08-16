// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"

#include "onnxruntime_cxx_api.h"

#include "test/show_ep_devices/printer.h"

namespace fs = std::filesystem;

ABSL_FLAG(std::string, plugin_ep_libs, "",
          "Specifies a list of plugin execution provider (EP) registration names and their corresponding shared libraries to register.\n"
          "[Usage]: --plugin_ep_libs \"plugin_ep_name_1|plugin_ep_1.dll plugin_ep_name_2|plugin_ep_2.dll ... \"");

ABSL_FLAG(std::string, format, "txt", "Specifies the output format. Valid formats are \"txt\" or \"json\".");

namespace onnxruntime::show_ep_devices {

namespace {

struct PluginEpRegistrationInfo {
  std::string registration_name;
  fs::path library_path;
};

std::vector<PluginEpRegistrationInfo> ParsePluginEpRegistrationInfo(absl::string_view plugin_ep_libs_str) {
  std::vector<PluginEpRegistrationInfo> result{};

  std::vector<absl::string_view> name_and_path_strs = absl::StrSplit(plugin_ep_libs_str, ' ', absl::SkipEmpty());
  result.reserve(name_and_path_strs.size());

  for (auto name_and_path_str : name_and_path_strs) {
    std::pair<absl::string_view, absl::string_view> name_and_path =
        absl::StrSplit(name_and_path_str, absl::MaxSplits('|', 1));
    const auto& [name, path] = name_and_path;

    PluginEpRegistrationInfo registration_info{};
    registration_info.registration_name = name;
    registration_info.library_path = path;
    result.emplace_back(std::move(registration_info));
  }

  return result;
}

using EpLibraryRegistrationHandle = std::unique_ptr<void, std::function<void(void*)>>;

EpLibraryRegistrationHandle RegisterEpLibrary(Ort::Env& env, const PluginEpRegistrationInfo& registration_info) {
  auto unregister_ep_library = [&env, registration_name = registration_info.registration_name](void*) {
    try {
      env.UnregisterExecutionProviderLibrary(registration_name.c_str());
    } catch (const Ort::Exception& e) {
      std::cerr << "Ort::Env::UnregisterExecutionProviderLibrary() failed: " << e.what() << "\n";
    }
  };

  env.RegisterExecutionProviderLibrary(registration_info.registration_name.c_str(),
                                       registration_info.library_path.native());

  // Set `handle_value` to something not equal to nullptr. The particular value doesn't really matter.
  // We are just using the unique_ptr deleter to unregister the EP library.
  void* const handle_value = reinterpret_cast<void*>(0x1);
  return EpLibraryRegistrationHandle{handle_value, unregister_ep_library};
}

std::vector<EpLibraryRegistrationHandle> RegisterEpLibraries(
    Ort::Env& env, const std::vector<PluginEpRegistrationInfo>& registration_infos) {
  std::vector<EpLibraryRegistrationHandle> result{};
  result.reserve(registration_infos.size());
  for (const auto& registration_info : registration_infos) {
    auto registration_handle = RegisterEpLibrary(env, registration_info);
    result.emplace_back(std::move(registration_handle));
  }
  return result;
}

}  // namespace

}  // namespace onnxruntime::show_ep_devices

int main(int argc, char** argv) {
  using namespace onnxruntime::show_ep_devices;

  try {
    absl::ParseCommandLine(argc, argv);

    const auto registration_infos = ParsePluginEpRegistrationInfo(absl::GetFlag(FLAGS_plugin_ep_libs));

    const auto output_format = ParseOutputFormat(absl::GetFlag(FLAGS_format));
    if (!output_format.has_value()) {
      throw std::invalid_argument("Invalid output format: " + absl::GetFlag(FLAGS_format));
    }

    Ort::Env env{};

    const auto registration_handles = RegisterEpLibraries(env, registration_infos);

    const auto ep_devices = env.GetEpDevices();

    PrintEpDeviceInfo(ep_devices, *output_format, std::cout);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
