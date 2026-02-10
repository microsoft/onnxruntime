// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/telum/telum_provider_factory_creator.h"

#include <cctype>

#include "core/common/common.h"
#include "core/providers/telum/telum_execution_provider.h"

namespace onnxruntime {
namespace {

// Case-insensitive "true/false/1/0" parsing. Keep intentionally strict to fail fast on typos.
bool ParseBool(std::string value) {
  for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  if (value == "1" || value == "true") return true;
  if (value == "0" || value == "false") return false;

  ORT_THROW("Invalid boolean value: '", value, "'. Expected one of: true, false, 1, 0.");
}

size_t ParseSizeT(const std::string& value) {
  long long v = 0;
  try {
    v = std::stoll(value);
  } catch (const std::exception&) {
    ORT_THROW("Invalid integer value: '", value, "'.");
  }

  if (v < 0) {
    ORT_THROW("Invalid integer value: '", value, "'. Must be non-negative.");
  }

  return static_cast<size_t>(v);
}

telum::TelumExecutionProviderInfo ParseTelumProviderOptions(const ProviderOptions& provider_options) {
  telum::TelumExecutionProviderInfo info{};

  for (const auto& kv : provider_options) {
    const auto& key = kv.first;
    const auto& value = kv.second;

    if (key == "strict_mode") {
      info.strict_mode = ParseBool(value);
    } else if (key == "enable_fusion") {
      info.enable_fusion = ParseBool(value);
    } else if (key == "log_fallbacks") {
      info.log_fallbacks = ParseBool(value);
    } else if (key == "max_batch_size") {
      info.max_batch_size = ParseSizeT(value);
    } else if (key == "max_sequence_length") {
      info.max_sequence_length = ParseSizeT(value);
    } else if (key == "create_arena") {
      info.create_arena = ParseBool(value);
    } else {
      // Ignore unknown keys for forward compatibility.
      // If we later want strict validation, flip this to ORT_THROW.
      continue;
    }
  }

  return info;
}

struct TelumProviderFactory final : IExecutionProviderFactory {
  explicit TelumProviderFactory(const ProviderOptions& provider_options)
      : info_{ParseTelumProviderOptions(provider_options)} {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<telum::TelumExecutionProvider>(info_);
  }

 private:
  telum::TelumExecutionProviderInfo info_;
};

}  // namespace

std::shared_ptr<IExecutionProviderFactory> TelumProviderFactoryCreator::Create(
    const ProviderOptions& provider_options, const SessionOptions* /*session_options*/) {
  return std::make_shared<TelumProviderFactory>(provider_options);
}

}  // namespace onnxruntime

