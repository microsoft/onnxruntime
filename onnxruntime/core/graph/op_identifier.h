// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ostream>
#include <string>
#include <string_view>
#include <tuple>

#include "core/common/hash_combine.h"

namespace onnxruntime {

template <typename StringType>
struct BasicOpIdentifier {
  using string_type = StringType;

  string_type domain;
  string_type op_type;
  int since_version;

  friend constexpr bool operator<(const BasicOpIdentifier<StringType>& lhs,
                                  const BasicOpIdentifier<StringType>& rhs) {
    return lhs.Tied() < rhs.Tied();
  }

  friend constexpr bool operator==(const BasicOpIdentifier<StringType>& lhs,
                                   const BasicOpIdentifier<StringType>& rhs) {
    return lhs.Tied() == rhs.Tied();
  }

  size_t GetHash() const {
    size_t h = std::hash<string_type>{}(domain);
    HashCombine(op_type, h);
    HashCombine(since_version, h);
    return h;
  }

  friend std::ostream& operator<<(std::ostream& os, const BasicOpIdentifier<StringType>& op_id) {
    os << op_id.domain << ':' << op_id.op_type << ':' << op_id.since_version;
    return os;
  }

 private:
  constexpr std::tuple<const string_type&, const string_type&, const int&> Tied() const {
    return std::tie(domain, op_type, since_version);
  }
};

using OpIdentifier = BasicOpIdentifier<std::string>;
using OpIdentifierWithStringViews = BasicOpIdentifier<std::string_view>;

}  // namespace onnxruntime

// add std::hash specializations
namespace std {
template <>
struct hash<onnxruntime::OpIdentifier> {
  size_t operator()(const onnxruntime::OpIdentifier& v) const {
    return v.GetHash();
  }
};

template <>
struct hash<onnxruntime::OpIdentifierWithStringViews> {
  size_t operator()(const onnxruntime::OpIdentifierWithStringViews& v) const {
    return v.GetHash();
  }
};
}  // namespace std
