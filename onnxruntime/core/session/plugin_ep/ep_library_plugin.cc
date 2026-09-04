// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_plugin.h"

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/session/environment.h"

namespace onnxruntime {
Status EpLibraryPlugin::Load() {
  auto status = Status::OK();

  ORT_TRY {
    std::lock_guard<std::mutex> lock{mutex_};
    if (factories_.empty()) {
      // Run the load steps in an inner lambda so that an early error return from ORT_RETURN_IF_ERROR
      // is captured in `status` rather than returning out of this function. That lets the cleanup
      // below run on ANY failure path (error status or exception). Previously the ORT_RETURN_IF_ERROR
      // returns bypassed the ORT_CATCH cleanup, so a partial load - e.g. the library handle was
      // acquired but a required export was missing or CreateEpFactories failed - left the library
      // loaded and half-initialized.
      status = [&]() -> Status {
        ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(library_path_, false, &handle_));
        ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "CreateEpFactories",
                                                                reinterpret_cast<void**>(&create_fn_)));
        ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "ReleaseEpFactory",
                                                                reinterpret_cast<void**>(&release_fn_)));

        // allocate buffer for EP to add factories to. library can add up to 4 factories.
        std::vector<OrtEpFactory*> factories{4, nullptr};

        size_t num_factories = 0;
        ORT_RETURN_IF_ERROR(ToStatusAndRelease(create_fn_(registration_name_.c_str(), OrtGetApiBase(),
                                                          logging::LoggingManager::DefaultLogger().ToExternal(),
                                                          factories.data(), factories.size(), &num_factories)));

        for (size_t i = 0; i < num_factories; ++i) {
          factories_.push_back(factories[i]);
        }

        return Status::OK();
      }();
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      // TODO: Add logging of exception
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load execution provider library: ", library_path_,
                               " with error: ", ex.what());
    });
  }

  // If the load failed for any reason (error status or exception), unload the library so a partial
  // load does not leave it loaded and half-initialized.
  if (!status.IsOK()) {
    auto unload_status = Unload();
    if (!unload_status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Failed to unload execution provider library: " << library_path_ << " with error: "
                          << unload_status.ErrorMessage();
    }
  }

  return status;
}

Status EpLibraryPlugin::Unload() {
  std::lock_guard<std::mutex> lock{mutex_};

  // Call ReleaseEpFactory for all factories and unload the library.
  // Current implementation assumes any error is permanent so does not leave pieces around to re-attempt Unload.
  if (handle_) {
    if (!factories_.empty()) {
      try {
        for (size_t idx = 0, end = factories_.size(); idx < end; ++idx) {
          auto* factory = factories_[idx];
          if (factory == nullptr) {
            continue;
          }

          auto status = ToStatusAndRelease(release_fn_(factory));
          if (!status.IsOK()) {
            // log it and treat it as released
            LOGS_DEFAULT(ERROR) << "ReleaseEpFactory failed for: " << library_path_ << " with error: "
                                << status.ErrorMessage();
          }

          factories_[idx] = nullptr;  // clear the pointer in case there's a failure before all are released
        }

        factories_.clear();
      } catch (const std::exception& ex) {
        LOGS_DEFAULT(ERROR) << "Failed releasing EP factories from " << library_path_ << ": " << ex.what();
      }
    }

    // TODO: Is there a better way? Is it worth worrying about?
    if (!factories_.empty()) {
      LOGS_DEFAULT(ERROR) << "Unloading " << library_path_ << ". " << factories_.size()
                          << " factories were not released due to errors. This may cause memory leaks. "
                             "Please check the error details in the log.";
    }

    ORT_RETURN_IF_ERROR(Env::Default().UnloadDynamicLibrary(handle_));
  }

  handle_ = nullptr;

  return Status::OK();
}
}  // namespace onnxruntime
