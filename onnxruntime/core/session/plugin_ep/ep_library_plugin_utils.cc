// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_plugin_utils.h"

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {
namespace ep_library_plugin_utils {

Status CreateFactories(CreateEpApiFactoriesFn create_fn, const std::string& registration_name,
                       std::vector<OrtEpFactory*>& factories) {
  // allocate buffer for EP to add factories to. library can add up to 4 factories.
  std::vector<OrtEpFactory*> new_factories{4, nullptr};

  size_t num_factories = 0;
  ORT_RETURN_IF_ERROR(ToStatusAndRelease(create_fn(registration_name.c_str(), OrtGetApiBase(),
                                                   logging::LoggingManager::DefaultLogger().ToExternal(),
                                                   new_factories.data(), new_factories.size(), &num_factories)));

  factories.reserve(factories.size() + num_factories);
  for (size_t i = 0; i < num_factories; ++i) {
    factories.push_back(new_factories[i]);
  }

  return Status::OK();
}

void ReleaseFactories(ReleaseEpApiFactoryFn release_fn, std::vector<OrtEpFactory*>& factories,
                      std::string_view library_description) {
  if (factories.empty()) {
    return;
  }

  try {
    for (size_t idx = 0, end = factories.size(); idx < end; ++idx) {
      auto* factory = factories[idx];
      if (factory == nullptr) {
        continue;
      }

      auto status = ToStatusAndRelease(release_fn(factory));
      if (!status.IsOK()) {
        // log it and treat it as released
        LOGS_DEFAULT(ERROR) << "ReleaseEpFactory failed for: " << library_description << " with error: "
                            << status.ErrorMessage();
      }

      factories[idx] = nullptr;  // clear the pointer in case there's a failure before all are released
    }

    factories.clear();
  } catch (const std::exception& ex) {
    LOGS_DEFAULT(ERROR) << "Failed releasing EP factories from " << library_description << ": " << ex.what();
  }

  // TODO: Is there a better way? Is it worth worrying about?
  if (!factories.empty()) {
    LOGS_DEFAULT(ERROR) << "Unloading " << library_description << ". " << factories.size()
                        << " factories were not released due to errors. This may cause memory leaks. "
                           "Please check the error details in the log.";
  }
}

}  // namespace ep_library_plugin_utils
}  // namespace onnxruntime
