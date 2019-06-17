// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace server {

namespace logging = logging;

class ServerEnvironment {
 public:
  explicit ServerEnvironment(logging::Severity severity, logging::LoggingManager::InstanceType instance_type = logging::LoggingManager::Default);
  ~ServerEnvironment() = default;
  ServerEnvironment(const ServerEnvironment&) = delete;

  const logging::Logger& GetAppLogger() const;
  std::unique_ptr<logging::Logger> GetLogger(const std::string& id);
  logging::Severity GetLogSeverity() const;

  Ort::Session& GetSession() const;
  common::Status InitializeModel(const std::string& model_path);
  const std::vector<std::string>& GetModelOutputNames() const;


 private:
  const logging::Severity severity_;
  const std::string logger_id_;
  logging::LoggingManager default_logging_manager_;

  Ort::Env runtime_environment_;
  Ort::SessionOptions options_;
  Ort::Session session;
  std::vector<std::string> model_output_names_;
};

}  // namespace server
}  // namespace onnxruntime
