// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <ranges>
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
  virtual ~EpLibrary() = default;

  virtual const char* RegistrationName() const = 0;
  virtual const std::filesystem::path* LibraryPath() const { return nullptr; }
  virtual Status Load() { return Status::OK(); }
  virtual Status Unload() { return Status::OK(); }

  std::ranges::view auto GetFactories() const {
    return std::views::iota(size_t{0}, GetFactoryCount()) |
           std::views::transform([this](size_t index) { return GetFactory(index); });
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibrary);

 private:
  virtual size_t GetFactoryCount() const = 0;
  virtual OrtEpFactory* GetFactory(size_t index) const = 0;
};
}  // namespace onnxruntime
