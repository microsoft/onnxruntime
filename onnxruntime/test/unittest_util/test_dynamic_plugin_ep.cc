// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/unittest_util/test_dynamic_plugin_ep.h"

#include <iostream>
#include <functional>
#include <memory>
#include <optional>

#include "nlohmann/json.hpp"

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_env.h"
#include "core/session/utils.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test::dynamic_plugin_ep_infra {

namespace {

using PluginEpLibraryRegistrationHandle = std::unique_ptr<void, std::function<void(void*)>>;

PluginEpLibraryRegistrationHandle RegisterPluginEpLibrary(Ort::Env& env,
                                                          const std::string& ep_library_registration_name,
                                                          const std::basic_string<ORTCHAR_T>& ep_library_path) {
  env.RegisterExecutionProviderLibrary(ep_library_registration_name.c_str(), ep_library_path);

  auto unregister_ep_library = [&env, registration_name = ep_library_registration_name](void* p) {
    if (p == nullptr) {
      return;
    }

    ORT_TRY {
      env.UnregisterExecutionProviderLibrary(registration_name.c_str());
    }
    ORT_CATCH(const Ort::Exception& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::cerr << "Failed to unregister EP library with name '" << registration_name << "': " << e.what() << "\n";
      });
    }
  };

  // Set `handle_value` to something not equal to nullptr. The particular value doesn't really matter.
  // We are just using the unique_ptr deleter to unregister the EP library.
  void* const handle_value = reinterpret_cast<void*>(0x1);
  return PluginEpLibraryRegistrationHandle{handle_value, unregister_ep_library};
}

void StrMapToKeyValueCstrVectors(const std::map<std::string, std::string>& m,
                                 std::vector<const char*>& keys_out, std::vector<const char*>& values_out) {
  std::vector<const char*> keys, values{};
  keys.reserve(m.size());
  values.reserve(m.size());
  for (auto& [key, value] : m) {
    keys.push_back(key.c_str());
    values.push_back(value.c_str());
  }
  keys_out = std::move(keys);
  values_out = std::move(values);
}

struct PluginEpInfrastructureState {
  InitializationConfig config{};
  PluginEpLibraryRegistrationHandle plugin_ep_library_registration_handle{};
  std::unique_ptr<IExecutionProviderFactory> ep_factory{};
  std::vector<const OrtEpDevice*> selected_c_ep_devices{};
  std::string ep_name{};
};

std::optional<PluginEpInfrastructureState> g_plugin_ep_infrastructure_state{};

}  // namespace

Status ParseInitializationConfig(std::string_view json_str, InitializationConfig& config_out) {
  using json = nlohmann::json;
  Status status = Status::OK();
  ORT_TRY {
    InitializationConfig config{};
    const auto parsed_json = json::parse(json_str);

    // required keys
    parsed_json.at("ep_library_registration_name").get_to(config.ep_library_registration_name);
    parsed_json.at("ep_library_path").get_to(config.ep_library_path);

    // optional keys
    config.default_ep_options = parsed_json.value<decltype(config.default_ep_options)>("default_ep_options", {});
    config.selected_ep_name = parsed_json.value<decltype(config.selected_ep_name)>("selected_ep_name", {});
    config.selected_ep_device_indices =
        parsed_json.value<decltype(config.selected_ep_device_indices)>("selected_ep_device_indices", {});

    config_out = std::move(config);
    return Status::OK();
  }
  ORT_CATCH(const json::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      constexpr std::string_view kExampleValidJsonStr =
          "{\n"
          "  \"ep_library_registration_name\": \"example_plugin_ep\",\n"
          "  \"ep_library_path\": \"/path/to/example_plugin_ep.dll\",\n"
          "  \"selected_ep_name\": \"example_plugin_ep\"\n"
          "}";

      status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "JSON parse error: ", e.what(),
                               "\nThis is an example valid JSON configuration:\n", kExampleValidJsonStr);
    });
  }
  return status;
}

Status Initialize(Ort::Env& env, InitializationConfig config) {
  ORT_RETURN_IF(IsInitialized(), "Already initialized.");

  auto ep_library_registration_handle = RegisterPluginEpLibrary(env, config.ep_library_registration_name,
                                                                ToPathString(config.ep_library_path));

  ORT_RETURN_IF(config.selected_ep_device_indices.empty() == config.selected_ep_name.empty(),
                "Exactly one of selected_ep_device_indices or selected_ep_name should be specified.");

  const auto ep_devices = env.GetEpDevices();
  std::vector<const OrtEpDevice*> selected_c_ep_devices{};

  if (!config.selected_ep_device_indices.empty()) {
    for (const auto idx : config.selected_ep_device_indices) {
      ORT_RETURN_IF(idx >= ep_devices.size(), "Selected EP device index is out of range: ", idx);
      selected_c_ep_devices.push_back(ep_devices[idx]);
    }
  } else {
    std::copy_if(ep_devices.begin(), ep_devices.end(), std::back_inserter(selected_c_ep_devices),
                 [&selected_ep_name = std::as_const(config.selected_ep_name)](Ort::ConstEpDevice ep_device) {
                   return ep_device.EpName() == selected_ep_name;
                 });
  }

  ORT_RETURN_IF(selected_c_ep_devices.empty(), "No EP devices were selected.");

  std::unique_ptr<IExecutionProviderFactory> ep_factory{};
  ORT_RETURN_IF_ERROR(
      CreateIExecutionProviderFactoryForEpDevices(static_cast<OrtEnv*>(env)->GetEnvironment(),
                                                  selected_c_ep_devices,
                                                  ep_factory));

  // Note: CreateIExecutionProviderFactoryForEpDevices() ensures that all EP devices refer to the same EP, so we will
  // just get the EP name from the first one.
  std::string ep_name = Ort::ConstEpDevice{selected_c_ep_devices.front()}.EpName();

  auto state = PluginEpInfrastructureState{};
  state.config = std::move(config);
  state.plugin_ep_library_registration_handle = std::move(ep_library_registration_handle);
  state.ep_factory = std::move(ep_factory);
  state.selected_c_ep_devices = std::move(selected_c_ep_devices);
  state.ep_name = std::move(ep_name);

  g_plugin_ep_infrastructure_state = std::move(state);
  return Status::OK();
}

bool IsInitialized() {
  return g_plugin_ep_infrastructure_state.has_value();
}

void Shutdown() {
  g_plugin_ep_infrastructure_state.reset();
}

std::unique_ptr<IExecutionProvider> MakeEp(const logging::Logger* logger) {
  if (!IsInitialized()) {
    return nullptr;
  }

  if (logger == nullptr) {
    logger = &DefaultLoggingManager().DefaultLogger();
  }

  const auto& state = *g_plugin_ep_infrastructure_state;

  std::vector<const char*> default_ep_option_key_cstrs{}, default_ep_option_value_cstrs{};
  StrMapToKeyValueCstrVectors(state.config.default_ep_options,
                              default_ep_option_key_cstrs, default_ep_option_value_cstrs);

  OrtSessionOptions ort_session_options{};
  ORT_THROW_IF_ERROR(AddEpOptionsToSessionOptions(state.selected_c_ep_devices,
                                                  default_ep_option_key_cstrs,
                                                  default_ep_option_value_cstrs,
                                                  ort_session_options.value));

  return state.ep_factory->CreateProvider(ort_session_options, *logger->ToExternal());
}

std::optional<std::string> GetEpName() {
  if (!IsInitialized()) {
    return std::nullopt;
  }

  return g_plugin_ep_infrastructure_state->ep_name;
}

}  // namespace onnxruntime::test::dynamic_plugin_ep_infra
