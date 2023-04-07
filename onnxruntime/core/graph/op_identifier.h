// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ostream>
#include <string>
#include <string_view>
#include <tuple>

#include "core/common/common.h"
#include "core/common/hash_combine.h"
#include "core/common/status.h"
#include "core/common/string_utils.h"
#include "core/common/parse_string.h"

namespace onnxruntime {

template <typename StringType>
struct BasicOpIdentifier {
  StringType domain;
  StringType op_type;
  int since_version;

  // comparison

  friend constexpr bool operator<(const BasicOpIdentifier& lhs,
                                  const BasicOpIdentifier& rhs) {
    return lhs.Tied() < rhs.Tied();
  }

  friend constexpr bool operator==(const BasicOpIdentifier& lhs,
                                   const BasicOpIdentifier& rhs) {
    return lhs.Tied() == rhs.Tied();
  }

  // hash computation

  size_t GetHash() const {
    size_t h = std::hash<StringType>{}(domain);
    HashCombine(op_type, h);
    HashCombine(since_version, h);
    return h;
  }

  // string conversion

  std::string ToString() const {
    return MakeString(domain, kStringRepresentationDelimiter,
                      op_type, kStringRepresentationDelimiter,
                      since_version);
  }

  static Status LoadFromString(std::string_view op_id_str, BasicOpIdentifier& op_id) {
    const auto components = utils::SplitString(op_id_str, kStringRepresentationDelimiter, true);
    ORT_RETURN_IF_NOT(components.size() == 3, "Invalid OpIdentifier string: ", op_id_str);
    int since_version{};
    ORT_RETURN_IF_NOT(TryParseStringWithClassicLocale(components[2], since_version),
                      "Failed to parse since_version from ", components[2]);
    op_id = BasicOpIdentifier{StringType{components[0]}, StringType{components[1]}, since_version};
    return Status::OK();
  }

  friend std::ostream& operator<<(std::ostream& os, const BasicOpIdentifier& op_id) {
    os << op_id.ToString();
    return os;
  }

 private:
  constexpr auto Tied() const {
    return std::tie(domain, op_type, since_version);
  }

  static constexpr std::string_view kStringRepresentationDelimiter = ":";
};

using OpIdentifier = BasicOpIdentifier<std::string>;

// An op identifier that uses std::string_view to refer to domain and op type values.
// IMPORTANT: Be sure that the underlying strings remain valid for the lifetime of the op identifier.
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
