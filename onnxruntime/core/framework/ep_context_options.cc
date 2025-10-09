// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <limits>
#include <string>
#include <utility>
#include "core/common/common.h"
#include "core/framework/ep_context_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace epctx {
// class ModelGenOptions

ModelGenOptions::ModelGenOptions() = default;

ModelGenOptions::ModelGenOptions(const ConfigOptions& config_options) {
  enable = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";

  std::string output_model_path = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
  if (!output_model_path.empty()) {
    output_model_location = std::filesystem::path(output_model_path);
  } else {
    output_model_location = std::monostate{};
  }

  std::string external_initializers_file_path = config_options.GetConfigOrDefault(
      kOrtSessionOptionsEpContextModelExternalInitializersFileName, "");
  if (!external_initializers_file_path.empty()) {
    ExternalInitializerFileInfo ext_info = {};
    ext_info.file_path = external_initializers_file_path;
    ext_info.size_threshold = 0;
    initializers_location = std::move(ext_info);
  }

  embed_ep_context_in_model = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
}

bool ModelGenOptions::HasOutputModelLocation() const {
  return !std::holds_alternative<std::monostate>(output_model_location);
}

const std::filesystem::path* ModelGenOptions::TryGetOutputModelPath() const {
  return std::get_if<std::filesystem::path>(&output_model_location);
}

const BufferHolder* ModelGenOptions::TryGetOutputModelBuffer() const {
  return std::get_if<BufferHolder>(&output_model_location);
}

const BufferWriteFuncHolder* ModelGenOptions::TryGetOutputModelWriteFunc() const {
  return std::get_if<BufferWriteFuncHolder>(&output_model_location);
}

bool ModelGenOptions::AreInitializersEmbeddedInOutputModel() const {
  return std::holds_alternative<std::monostate>(initializers_location);
}

const ExternalInitializerFileInfo* ModelGenOptions::TryGetExternalInitializerFileInfo() const {
  return std::get_if<ExternalInitializerFileInfo>(&initializers_location);
}

const InitializerHandler* ModelGenOptions::TryGetInitializerHandler() const {
  return std::get_if<InitializerHandler>(&initializers_location);
}

}  // namespace epctx
}  // namespace onnxruntime
