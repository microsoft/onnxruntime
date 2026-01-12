// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "../plugin_ep_utils.h"
#include "ep_factory.h"

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

static std::string GetLowercaseString(std::string str) {
  // https://en.cppreference.com/w/cpp/string/byte/tolower
  // The behavior of tolower from <cctype> is undefined if the argument is neither representable as unsigned char
  // nor equal to EOF. To use tolower safely with a plain char (or signed char), the argument must be converted to
  // unsigned char.
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return str;
}

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           const OrtLogger* default_logger,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Manual init for the C++ API
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  OrtKeyValuePairs* c_kvps = nullptr;
  RETURN_IF_ERROR(ep_api->GetEnvConfigEntries(&c_kvps));
  Ort::KeyValuePairs env_configs(c_kvps);

  // Note: environment configuration entries for a specific EP library are stored with the prefix:
  // 'ep_lib.<lower_case_ep_lib_registration_name>.'.
  // Here we extract a config that determines whether creating virtual hardware devices is allowed.
  std::string config_key = "ep_lib.";
  config_key += GetLowercaseString(registration_name);
  config_key += ".allow_virtual_devices";
  const char* config_value = env_configs.GetValue(config_key.c_str());
  const bool allow_virtual_devices = config_value != nullptr && strcmp(config_value, "1") == 0;

  std::unique_ptr<OrtEpFactory> factory = std::make_unique<EpFactoryVirtualGpu>(*ort_api, *ep_api, *model_editor_api,
                                                                                allow_virtual_devices, *default_logger);

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<EpFactoryVirtualGpu*>(factory);
  return nullptr;
}

}  // extern "C"
