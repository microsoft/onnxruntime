// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_plugin_utils.h"

#include <array>
#include <iterator>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {
namespace ep_library_plugin_utils {

void OrtEpFactoryDeleter::operator()(OrtEpFactory* factory) const noexcept {
  if (factory == nullptr) {
    return;
  }

  auto status = ToStatusAndRelease(release_fn(factory));
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "ReleaseEpFactory failed with error: " << status.ErrorMessage();
  }
}

Status CreateFactories(CreateEpApiFactoriesFn create_fn, ReleaseEpApiFactoryFn release_fn,
                       const std::string& registration_name,
                       std::vector<OrtEpFactoryUniquePtr>& factories) {
  constexpr size_t kMaxFactories = 4;
  std::array<OrtEpFactory*, kMaxFactories> raw_factories{};
  std::array<OrtEpFactoryUniquePtr, kMaxFactories> new_factories{};

  // Make adoption non-throwing before provider code transfers any ownership to us.
  const auto adopt_factories = [&]() noexcept {
    for (size_t i = 0; i < raw_factories.size(); ++i) {
      if (raw_factories[i] != nullptr) {
        new_factories[i] = OrtEpFactoryUniquePtr{raw_factories[i], OrtEpFactoryDeleter{release_fn}};
        raw_factories[i] = nullptr;
      }
    }
  };

  size_t num_factories = 0;
  OrtStatus* ort_status = create_fn(registration_name.c_str(), OrtGetApiBase(),
                                    logging::LoggingManager::DefaultLogger().ToExternal(),
                                    raw_factories.data(), raw_factories.size(), &num_factories);
  adopt_factories();

  if (ort_status != nullptr) {
    return ToStatusAndRelease(ort_status);
  }

  ORT_RETURN_IF_NOT(num_factories <= new_factories.size(),
                    "CreateEpFactories returned ", num_factories, " factories but the supplied capacity was ",
                    new_factories.size(), ".");

  for (size_t i = 0; i < new_factories.size(); ++i) {
    if ((i < num_factories) != (new_factories[i] != nullptr)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "CreateEpFactories returned factory outputs inconsistent with num_factories.");
    }
  }

  factories.reserve(SafeInt<size_t>(factories.size()) + num_factories);
  factories.insert(factories.end(),
                   std::make_move_iterator(new_factories.begin()),
                   std::make_move_iterator(new_factories.begin() + num_factories));

  return Status::OK();
}

}  // namespace ep_library_plugin_utils
}  // namespace onnxruntime
