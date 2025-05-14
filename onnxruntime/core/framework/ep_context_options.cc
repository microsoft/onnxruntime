// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ep_context_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace epctx {
ModelGenOptions::ModelGenOptions(const ConfigOptions& config_options) {
  enable = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
  output_model_location = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
  output_external_initializers_file_path = config_options.GetConfigOrDefault(
      kOrtSessionOptionsEpContextModelExternalInitializersFileName, "");
  output_external_initializer_size_threshold = 0;
  embed_ep_context_in_model = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
}

bool ModelGenOptions::HasOutputModelLocation() const {
  return !std::holds_alternative<std::monostate>(output_model_location);
}

const std::string* ModelGenOptions::TryGetOutputModelPath() const {
  return std::get_if<std::string>(&output_model_location);
}

const BufferHolder* ModelGenOptions::TryGetOutputModelBuffer() const {
  return std::get_if<BufferHolder>(&output_model_location);
}

const StreamHolder* ModelGenOptions::TryGetOutputModelStream() const {
  return std::get_if<StreamHolder>(&output_model_location);
}

}  // namespace epctx
}  // namespace onnxruntime
