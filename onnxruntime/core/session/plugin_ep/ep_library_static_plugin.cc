// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_static_plugin.h"

#include "core/common/common.h"
#include "core/session/plugin_ep/ep_library_plugin_utils.h"

namespace onnxruntime {
Status EpLibraryStaticPlugin::Load() {
  auto status = Status::OK();

  ORT_TRY {
    std::lock_guard<std::mutex> lock{mutex_};
    if (factories_.empty()) {
      status = ep_library_plugin_utils::CreateFactories(create_fn_, registration_name_, factories_);
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to create factories for statically linked execution "
                               "provider '",
                               registration_name_, "' with error: ", ex.what());
    });
  }

  // Release anything that was created before the failure so a partial load does not leave the library
  // half-initialized. There is no library handle to release.
  if (!status.IsOK()) {
    std::lock_guard<std::mutex> lock{mutex_};
    ep_library_plugin_utils::ReleaseFactories(release_fn_, factories_, registration_name_);
  }

  return status;
}

Status EpLibraryStaticPlugin::Unload() {
  std::lock_guard<std::mutex> lock{mutex_};

  ep_library_plugin_utils::ReleaseFactories(release_fn_, factories_, registration_name_);

  return Status::OK();
}
}  // namespace onnxruntime
