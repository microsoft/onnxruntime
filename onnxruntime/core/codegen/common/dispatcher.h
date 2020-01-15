// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include <functional>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace codegen {

// DispatcherBase is a customized unordered_map
// that provides all codegen-related functionality
// including 1) dispatching a pass
//           2) dump corresponding name
// DispatcherBase may or may not keep ownership,
// depending on the template parameter, CONTENT_TYPE.
// Note DispatcherBase has a protected destructor

template <typename CONTENT_TYPE>
class DispatcherBase {
 public:
  DispatcherBase(const std::string& name)
      : name_(name) {}

  const std::string& Name() const {
    return name_;
  }

  bool Contains(const std::string& name) const {
    return contents_.count(name) > 0;
  }

  void ForEach(std::function<void(const std::string&,
                                  CONTENT_TYPE)>
                   func) {
    for (auto& p : contents_) {
      func(p.first, p.second);
    }
  }

  bool Register(const std::string& name,
                CONTENT_TYPE op) {
    if (!Contains(name)) {
      contents_.emplace(name, op);
      return true;
    }
    return false;
  }

  CONTENT_TYPE Get(const std::string& key) const {
    auto iter = contents_.find(key);
    if (iter != contents_.end()) {
      return iter->second;
    }
    return nullptr;
  }

  const std::unordered_map<std::string, CONTENT_TYPE> GetContents() const {
    return contents_;
  }

  std::unordered_map<std::string, CONTENT_TYPE> GetMutableContents() {
    return contents_;
  }

 protected:
  std::string name_;
  std::unordered_map<std::string, CONTENT_TYPE> contents_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DispatcherBase);
  ~DispatcherBase() = default;
};

}  // namespace codegen
}  // namespace onnxruntime
