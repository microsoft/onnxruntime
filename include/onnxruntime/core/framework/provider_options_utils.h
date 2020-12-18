// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
class ProviderOptionsParser {
 public:
  /**
   * Adds a parser for a particular provider option value.
   *
   * @param name The provider option name.
   * @param value_parser An object that parses the option value.
   *        It should be callable with the following signature and return
   *        whether the parsing was successful:
   *            Status value_parser(const std::string&)
   *
   * @return The current ProviderOptionsParser instance.
   */
  template <typename ValueParserType>
  ProviderOptionsParser& AddValueParser(
      const std::string& name, ValueParserType value_parser) {
    ORT_ENFORCE(
        value_parsers_.emplace(name, ValueParser{value_parser}).second,
        "Provider option \"", name, "\" already has a value parser.");
    return *this;
  }

  /**
   * Adds a parser for a particular provider option value which converts a
   * value to the right type and assigns it to the given reference.
   *
   * IMPORTANT: This function stores a reference to the destination variable.
   * The caller must ensure that the reference is valid when Parse() is called!
   *
   * @param name The provider option name.
   * @param dest The destination variable reference.
   *
   * @return The current ProviderOptionsParser instance.
   */
  template <typename ValueType>
  ProviderOptionsParser& AddAssignmentToReference(
      const std::string& name, ValueType& dest) {
    return AddValueParser(
        name,
        [&dest](const std::string& value_str) -> Status {
          return ParseString(value_str, dest);
        });
  }

  /**
   * Parses the given provider options.
   */
  Status Parse(const ProviderOptions& options) const {
    for (const auto& option : options) {
      const auto& name = option.first;
      const auto& value_str = option.second;

      const auto value_parser_it = value_parsers_.find(name);
      ORT_RETURN_IF(
          value_parser_it == value_parsers_.end(),
          "Unknown provider option: \"", name, "\".");

      const auto parse_status = value_parser_it->second(value_str);
      ORT_RETURN_IF_NOT(
          parse_status.IsOK(),
          "Failed to parse provider option \"", name, "\": ", parse_status.ErrorMessage());
    }

    return Status::OK();
  }

 private:
  using ValueParser = std::function<Status(const std::string&)>;
  std::unordered_map<std::string, ValueParser> value_parsers_;
};

}  // namespace onnxruntime
