// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

#include "core/session/onnxruntime_env_config_keys.h"

#include "../plugin_ep_utils.h"
#include "ep_factory.h"

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                                           const OrtLogger* default_logger,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Manual init for the C++ API
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  Ort::KeyValuePairs env_configs = Ort::GetEnvConfigEntries();

  // Extract a config that determines whether creating virtual hardware devices is allowed.
  // An application can allow an EP library to create virtual devices in two ways:
  //  1. Use an EP library registration name that ends in the suffix ".virtual". If so, ORT will automatically
  //     set the config key "allow_virtual_devices" to "1" in the environment.
  //  2. Directly set the config key "allow_virtual_devices" to "1" when creating the
  //     OrtEnv via OrtApi::CreateEnvWithOptions().
  const char* config_value = env_configs.GetValue(kOrtEnvAllowVirtualDevices);
  const bool allow_virtual_devices = config_value != nullptr && strcmp(config_value, "1") == 0;

  std::unique_ptr<OrtEpFactory> factory = std::make_unique<EpFactoryVirtualGpu>(*ort_api, *ep_api, *model_editor_api,
                                                                                allow_virtual_devices, *default_logger);

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<EpFactoryVirtualGpu*>(factory);
  return nullptr;
}

}  // extern "C"
