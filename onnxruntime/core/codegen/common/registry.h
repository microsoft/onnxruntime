// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include <functional>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace codegen {

// RegistryBase is a customized unordered_map
// that keep ownership of passes,
// including 1) IR builder passes
//           2) Weight layout transformer passes
//           3) Scheduler passses, etc.

template <typename CONTENT_TYPE>
class RegistryBase {
 public:
  RegistryBase() = default;

  virtual ~RegistryBase() = default;

  bool Contains(const std::string& name) const {
    return contents_.count(name) > 0;
  }

  CONTENT_TYPE* Get(const std::string& name) const {
    if (contents_.find(name) != contents_.end())
      return contents_.at(name).get();
    return nullptr;
  }

  CONTENT_TYPE* RegisterOrGet(
      const std::string& name,
      std::unique_ptr<CONTENT_TYPE>&& ptr) {
    if (!Contains(name))
      contents_.emplace(name, std::move(ptr));
    return Get(name);
  }

  CONTENT_TYPE* RegisterOrGet(
      std::unique_ptr<CONTENT_TYPE>&& ptr) {
    return RegisterOrGet(ptr->Name(), std::move(ptr));
  }

  bool Register(
      const std::string& name,
      std::unique_ptr<CONTENT_TYPE>&& ptr) {
    if (!Contains(name)) {
      contents_.emplace(name, std::move(ptr));
      return true;
    }
    return false;
  }

  bool Register(
      std::unique_ptr<CONTENT_TYPE>&& ptr) {
    return Register(ptr->Name(), std::move(ptr));
  }

 protected:
  std::unordered_map<std::string, std::unique_ptr<CONTENT_TYPE>> contents_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RegistryBase);
};

// Put common Registry Management utilities if these is any

}  // namespace codegen
}  // namespace onnxruntime
