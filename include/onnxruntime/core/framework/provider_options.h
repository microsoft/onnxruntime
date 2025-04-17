// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/string_utils.h"

namespace onnxruntime {

// data types for execution provider options

using ProviderOptions = std::unordered_map<std::string, std::string>;
using ProviderOptionsVector = std::vector<ProviderOptions>;
using ProviderOptionsMap = std::unordered_map<std::string, ProviderOptions>;

struct ProviderOptionsUtils {
  // return the prefix that is used for provider options when they're in SessionOptions.config_options.
  // the prefix is not used when ProviderOptions are coming from the user.
  static std::string GetProviderOptionPrefix(const std::string& name) {
    return GetProviderOptionPrefix(name.c_str());
  }

  // return the prefix that is used for provider options when they're in SessionOptions.config_options.
  // the prefix is not used when ProviderOptions are coming from the user.
  static std::string GetProviderOptionPrefix(const char* name) {
    return std::string("ep.") + utils::GetLowercaseString(name) + ".";
  }
};

}  // namespace onnxruntime
