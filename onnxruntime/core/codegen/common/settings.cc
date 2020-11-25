// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/common/settings.h"

#include "core/common/logging/logging.h"
#include <algorithm>
#include <cctype>

namespace onnxruntime {
namespace codegen {

CodeGenSettings& CodeGenSettings::Instance() {
  static CodeGenSettings settings;
  return settings;
}

CodeGenSettings::CodeGenSettings() {}

void CodeGenSettings::InsertOptions(const std::map<std::string, std::string>& options) {
  for (const auto& option : options) {
    const auto& key = option.first;
    const auto& value = option.second;

    auto iter = options_.find(key);
    // found existing ones
    if (iter != options_.end()) {
      if (iter->second != value) {
        LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "CodeGenSettings: option"
                                                 << key << " is overridded from: "
                                                 << iter->second << " to: " << value;
        iter->second = value;
      }
    } else {
      options_.insert(std::make_pair(key, value));
    }
  }
}

void CodeGenSettings::DumpOptions() const {
  std::ostringstream stream;
  stream << "CodeGenSettings: dump all options" << std::endl;
  for (const auto& option : options_) {
    stream << "  " << option.first << " = " << option.second << std::endl;
  }
  LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << stream.str();
}

std::string CodeGenSettings::GetOptionValue(const std::string& key) const {
  const auto& iter = options_.find(key);
  if (iter == options_.end()) {
    LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "CodeGenSettings::GetOptionValue: unrecognized option" << key;
    return "";
  }
  return iter->second;
}

bool CodeGenSettings::HasOption(const std::string& key) const {
  return options_.count(key) > 0;
}

bool CodeGenSettings::OptionMatches(const std::string& key, const std::string& value) const {
  if (!HasOption(key))
    return false;

#ifdef _WIN32
  return 0 == _stricmp(options_.at(key).c_str(), value.c_str());
#else
  return 0 == strcasecmp(options_.at(key).c_str(), value.c_str());
#endif
}

void CodeGenSettings::Clear() {
  options_.clear();
}

}  // namespace codegen
}  // namespace onnxruntime
