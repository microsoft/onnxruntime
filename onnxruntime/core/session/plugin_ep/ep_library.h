// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/status.h"
#include "core/framework/provider_options.h"
#include "core/framework/session_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

/// <summary>
/// EpLibrary is the base class for implementing support for execution provider libraries that provide
/// OrtEpFactory instances.
/// </summary>
class EpLibrary {
 public:
  EpLibrary() = default;

  virtual const char* RegistrationName() const = 0;
  virtual Status Load() { return Status::OK(); }
  virtual const std::vector<OrtEpFactory*>& GetFactories() = 0;  // valid after Load()
  virtual Status Unload() { return Status::OK(); }

  virtual ~EpLibrary() = default;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibrary);
};
}  // namespace onnxruntime
