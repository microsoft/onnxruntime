// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "core/graph/constants.h"
#include "core/framework/session_options.h"

namespace onnxruntime {
namespace qnnctxgen {

// Configuration for initializing the dynamic plugin EP infrastructure.
struct PluginEpConfig {
  std::string ep_library_registration_name{};
  std::string ep_library_path{};

  // Note: Exactly one of `selected_ep_name` or `selected_ep_device_indices` should be set.
  // An empty value for either means it is unset.

  // Specifies the EP devices matching this EP name as the selected EP devices.
  std::string selected_ep_name{};
  // Specifies the selected EP devices by their indices.
  std::vector<size_t> selected_ep_device_indices{};

  std::unordered_map<std::string, std::string> default_ep_options{};
};

struct MachineConfig {
  std::string provider_type_name{onnxruntime::kQnnExecutionProvider};
  std::optional<PluginEpConfig> plugin_ep_config = std::nullopt;
};

struct RunConfig {
  bool f_verbose{false};
  std::unordered_map<std::string, std::string> session_config_entries;
  std::unordered_map<std::string, std::string> provider_options;
};

struct TestConfig {
  std::vector<std::basic_string<ORTCHAR_T>> model_file_paths;
  RunConfig run_config;
  MachineConfig machine_config;
};

}  // namespace qnnctxgen
}  // namespace onnxruntime
