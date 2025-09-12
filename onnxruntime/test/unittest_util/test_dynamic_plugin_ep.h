// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "core/common/status.h"

namespace onnxruntime {
struct IExecutionProviderFactory;
class IExecutionProvider;

namespace logging {
class Logger;
}  // namespace logging

namespace test {

// `onnxruntime::test::dynamic_plugin_ep_infra` contains functions and types related to dynamically loaded plugin EP
// unit testing infrastructure.
namespace dynamic_plugin_ep_infra {

// Note: `Initialize()` and `Shutdown()` are not thread-safe.
// They should be called before and after calls to most of the other functions in this namespace.
// The exception to this is `ParseInitializationConfig()`, which may be called before `Initialize()`.

// Configuration for initializing the dynamic plugin EP infrastructure.
struct InitializationConfig {
  std::string ep_library_registration_name{};
  std::string ep_library_path{};

  // Note: Exactly one of `selected_ep_name` or `selected_ep_device_indices` should be set.
  // An empty value for either means it is unset.

  // Specifies the EP devices matching this EP name as the selected EP devices.
  std::string selected_ep_name{};
  // Specifies the selected EP devices by their indices.
  std::vector<size_t> selected_ep_device_indices{};

  std::map<std::string, std::string> default_ep_options{};
};

// Parses `InitializationConfig` from JSON.
// The configuration JSON object should have keys and values that match the `InitializationConfig` fields.
// E.g.:
// {
//   "ep_library_registration_name": "example_plugin_ep",
//   "ep_library_path": "/path/to/example_plugin_ep.dll",
//   "selected_ep_name": "example_plugin_ep",
//   "default_ep_options": { "option_key": "option_value" }
// }
Status ParseInitializationConfig(std::string_view json_str, InitializationConfig& config);

// Initializes dynamic plugin EP infrastructure.
Status Initialize(Ort::Env& env, InitializationConfig config);

// Gets whether the dynamic plugin EP infrastructure is initialized.
bool IsInitialized();

// Shuts down dynamic plugin EP infrastructure.
// This does not require a previously successful call to `Initialize()`.
void Shutdown();

// Returns a dynamic plugin EP `IExecutionProvider` instance, or `nullptr` if uninitialized.
std::unique_ptr<IExecutionProvider> MakeEp(const logging::Logger* logger = nullptr);

// Gets the dynamic plugin EP name, or `std::nullopt` if uninitialized.
std::optional<std::string> GetEpName();

}  // namespace dynamic_plugin_ep_infra
}  // namespace test
}  // namespace onnxruntime
