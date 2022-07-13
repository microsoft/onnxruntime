#include "core/framework/ex_lib_loader.h"
#include "core/platform/env.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
namespace onnxruntime {
ExLibLoader::~ExLibLoader() {
  ORT_TRY {
    for (auto& elem : dso_name_data_map_) {
      LOGS_DEFAULT(INFO) << "Unloading DSO " << elem.first;

      PreUnloadLibrary(elem.second);

      // unload the DSO
      if (!Env::Default().UnloadDynamicLibrary(elem.second).IsOK()) {
        LOGS_DEFAULT(WARNING) << "Failed to unload DSO: " << elem.first;
      }
    }
  }
  ORT_CATCH(const std::exception& ex) {
    // make sure exceptions don't leave the destructor
    ORT_HANDLE_EXCEPTION([&ex]() {
      LOGS_DEFAULT(WARNING) << "Caught exception while destructing CustomOpsLoader with message: " << ex.what();
    });
  }
}

void* ExLibLoader::GetExLibHandle(const std::string& dso_file_path) const {
  auto it = dso_name_data_map_.find(dso_file_path);
  return it == dso_name_data_map_.end() ? nullptr : it->second;
}

common::Status ExLibLoader::LoadExternalLib(const std::string& dso_file_path,
                                            void** handle) {
  auto status = Status::OK();
  ORT_TRY {
    if (dso_name_data_map_.count(dso_file_path)) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "A dso with name " + dso_file_path + " has already been loaded.");
    }

    void* lib_handle = nullptr;
    ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(dso_file_path, false, &lib_handle));
    dso_name_data_map_[dso_file_path] = lib_handle;
    *handle = lib_handle;
    return Status::OK();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(common::ONNXRUNTIME, common::FAIL, "Caught exception while loading custom ops with message: " + std::string(ex.what()));
    });
  }

  return status;
}

}  // namespace onnxruntime
