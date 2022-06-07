// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <assert.h>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"

namespace onnxruntime {
namespace common {
template <class... Types>
class Record {
 public:
  typedef std::tuple<Types...> Values;

  Record() = default;

  Record(const std::vector<std::string>& names, const Values& values) {
    ORT_ENFORCE(std::tuple_size<Values>::value == names.size(),
                "Parameter sizes do not match. %d != %d", std::tuple_size<Values>::value, names.size());
    names_ = names;
    values_ = values;
  }

  Record(const Record<Types...>& other) {
    names_ = other.names_;
    values_ = other.values_;
  }

  Status GetName(size_t index, const std::string** pp_name) const {
    if (nullptr == pp_name || index >= names_.size()) {
      return Status(ONNXRUNTIME, common::INVALID_ARGUMENT);
    }

    *pp_name = &(names_[index]);
    return Status::OK();
  }

  const Values& GetValues() const {
    return values_;
  }

 private:
  std::vector<std::string> names_;

  Values values_;
};
}  // namespace common
}  // namespace onnxruntime
