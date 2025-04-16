// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/status.h"
#include "core/framework/provider_options.h"
#include "core/framework/session_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

struct EpLibrary {
  virtual const char* RegistrationName() const = 0;
  virtual Status Load() { return Status::OK(); }
  virtual const std::vector<OrtEpFactory*>& GetFactories() = 0;  // valid after Load()
  virtual Status Unload() { return Status::OK(); }
  virtual ~EpLibrary() = default;

 protected:
  static ProviderOptions GetOptionsFromSessionOptions(const std::string& ep_name,
                                                      const SessionOptions& session_options);
};
}  // namespace onnxruntime
