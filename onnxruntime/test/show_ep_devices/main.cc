// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/flags/usage_config.h"
#include "absl/strings/str_split.h"

#include "onnxruntime_cxx_api.h"

#include "test/show_ep_devices/printer.h"

// Exception macros to accomodate builds with exceptions disabled.

#if !defined(ORT_NO_EXCEPTIONS)
#define TRY try
#define CATCH(x) catch (x)
#define HANDLE_EXCEPTION(func) func()
#define THROW(e) throw(e)
#else  // defined(ORT_NO_EXCEPTIONS)
#define TRY if (true)
#define CATCH(x) else if (false)
#define HANDLE_EXCEPTION(func)
#define THROW(e)                                  \
  do {                                            \
    std::cerr << "Error: " << (e).what() << "\n"; \
    std::abort();                                 \
  } while (0)
#endif  // defined(ORT_NO_EXCEPTIONS)

// Command line options

ABSL_FLAG(std::string, plugin_ep_libs, "",
          "Specifies a list of plugin execution provider (EP) registration names and the corresponding shared "
          "library paths to register.\n"
          "Format: <registration_name>|<library_path>\n"
          "For example: "
          "--plugin_ep_libs \"plugin_ep_name_1|plugin_ep_1.dll plugin_ep_name_2|plugin_ep_2.dll ...\"");

ABSL_FLAG(std::string, format, "txt", "Specifies the output format. Valid formats are \"txt\" or \"json\".");

namespace fs = std::filesystem;

namespace onnxruntime::show_ep_devices {

namespace {

void InitializeAbslFlags(int argc, char** argv) {
  {
    absl::FlagsUsageConfig flags_usage_config{};
    flags_usage_config.contains_help_flags = [](absl::string_view filename) -> bool {
      const fs::path this_file{__FILE__}, file{filename};
      return file.filename() == this_file.filename();
    };
    absl::SetFlagsUsageConfig(std::move(flags_usage_config));
  }

  absl::SetProgramUsageMessage(
      "Shows information about the available EP devices on the current system. "
      "Use the --plugin_ep_libs option to specify plugin EP libraries to register.");

  absl::ParseCommandLine(argc, argv);
}

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
  auto unregister_ep_library = [&env, registration_name = registration_info.registration_name](void* p) {
    if (p == nullptr) {
      return;
    }

    TRY {
      env.UnregisterExecutionProviderLibrary(registration_name.c_str());
    }
    CATCH(const Ort::Exception& e) {
      HANDLE_EXCEPTION([&]() {
        std::cerr << "Ort::Env::UnregisterExecutionProviderLibrary() failed: " << e.what() << "\n";
      });
    }
  };

  // Note: Ort::Env::RegisterExecutionProviderLibrary() treats relative paths as relative to the directory containing
  // onnxruntime core runtime code (e.g., the onnxruntime shared library). That directory may be different from the
  // current directory, so we make relative paths absolute here in order for them to behave as expected (i.e., to be
  // relative to the current directory).
  const auto absolute_library_path = fs::absolute(registration_info.library_path);

  env.RegisterExecutionProviderLibrary(registration_info.registration_name.c_str(),
                                       absolute_library_path.native());

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
  int return_value = 0;
  TRY {
    using namespace onnxruntime::show_ep_devices;

    InitializeAbslFlags(argc, argv);

    const auto registration_infos = ParsePluginEpRegistrationInfo(absl::GetFlag(FLAGS_plugin_ep_libs));

    const auto output_format = ParseOutputFormat(absl::GetFlag(FLAGS_format));
    if (!output_format.has_value()) {
      THROW(std::invalid_argument{"Invalid output format: " + absl::GetFlag(FLAGS_format)});
    }

    Ort::Env env{};

    const auto registration_handles = RegisterEpLibraries(env, registration_infos);

    const auto ep_devices = env.GetEpDevices();

    PrintEpDeviceInfo(ep_devices, *output_format, std::cout);
  }
  CATCH(const std::exception& e) {
    HANDLE_EXCEPTION([&]() {
      std::cerr << "Error: " << e.what() << "\n";
      return_value = 1;
    });
  }

  return return_value;
}
