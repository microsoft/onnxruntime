// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
#include <regex>

#include "core/providers/stvm/stvm_execution_provider_info.h"

#include "core/common/common.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace stvm {
namespace provider_option_names {
constexpr const char* kTarget = "target";
constexpr const char* kTargetHost = "target_host";
constexpr const char* kOptLevel = "opt_level";
constexpr const char* kFreezeWeights = "freeze_weights";
constexpr const char* kToNHWC = "to_nhwc";
constexpr const char* kTuningFilePath = "tuning_file_path";
constexpr const char* kTuningType = "tuning_type";
constexpr const char* kInputNames = "input_names";
constexpr const char* kInputShapes = "input_shapes";

static const std::unordered_set<std::string> valid_keys {
  std::string{kTarget},
  std::string{kTargetHost},
  std::string{kOptLevel},
  std::string{kFreezeWeights},
  std::string{kToNHWC},
  std::string{kTuningFilePath},
  std::string{kTuningType},
  std::string{kInputNames},
  std::string{kInputShapes}
};

}  // namespace provider_option_names
}  // namespace stvm

std::string StvmExecutionProviderInfo::whitespace_trimming(const std::string& str) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  size_t start = str.find_first_not_of(WHITESPACE);
  if (start == std::string::npos) {
    return "";
  } else {
    size_t end = str.find_last_not_of(WHITESPACE);
    ORT_ENFORCE(end != std::string::npos);
    return str.substr(start, end + 1);
  }
}

StvmExecutionProviderInfo StvmExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  StvmExecutionProviderInfo info{};

  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddAssignmentToReference(stvm::provider_option_names::kTarget, info.target)
          .AddAssignmentToReference(stvm::provider_option_names::kTargetHost, info.target_host)
          .AddAssignmentToReference(stvm::provider_option_names::kOptLevel, info.opt_level)
          .AddAssignmentToReference(stvm::provider_option_names::kFreezeWeights, info.freeze_weights)
          .AddAssignmentToReference(stvm::provider_option_names::kToNHWC, info.to_nhwc)
          .AddAssignmentToReference(stvm::provider_option_names::kTuningFilePath, info.tuning_file_path)
          .AddAssignmentToReference(stvm::provider_option_names::kTuningType, info.tuning_type)
          .AddAssignmentToReference(stvm::provider_option_names::kInputNames, info.input_names_str)
          .AddAssignmentToReference(stvm::provider_option_names::kInputShapes, info.input_shapes_str)
          .Parse(options));

  return info;
}

StvmExecutionProviderInfo StvmExecutionProviderInfo::FromOptionsString(const char* opt_str) {
  std::string settings{opt_str};
  ProviderOptions options;
  if (!settings.empty()) {
    const std::string& str = settings;

    // tokenize settings
    std::regex reg("\\s*,\\s*");
    std::sregex_token_iterator iter(str.begin(), str.end(), reg, -1);
    std::sregex_token_iterator iter_end;
    std::vector<std::string> pairs(iter, iter_end);

    ORT_ENFORCE(pairs.size() > 0);

    for(const auto& pair : pairs) {
      auto pos_colon = pair.find(':');
      ORT_ENFORCE(pos_colon != std::string::npos, "Invalid key value pair.");
      std::string key = pair.substr(0, pos_colon);
      std::string value = pair.substr(pos_colon + 1);

      // trim leading and trailing spaces from key/value
      key = whitespace_trimming(key);
      value = whitespace_trimming(value);

      // Check keys of obtained options
      if (stvm::provider_option_names::valid_keys.count(key) == 0) {
        ORT_NOT_IMPLEMENTED("StvmOptions: unknown option (", key, ")");
      }

      options[key] = value;
    }
  }

  return FromProviderOptions(options);
}

}  // namespace onnxruntime
