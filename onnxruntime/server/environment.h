// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"
#include <spdlog/spdlog.h>

namespace onnxruntime {
namespace server {


class ServerEnvironment {
 public:
  explicit ServerEnvironment(OrtLoggingLevel severity, spdlog::sink_ptr sink);
  ~ServerEnvironment() = default;
  ServerEnvironment(const ServerEnvironment&) = delete;

  OrtLoggingLevel GetLogSeverity() const;

  const Ort::Session& GetSession() const;
  void InitializeModel(const std::string& model_path);
  const std::vector<std::string>& GetModelOutputNames() const;
  std::shared_ptr<spdlog::logger> GetLogger(const std::string& request_id) const;
  std::shared_ptr<spdlog::logger> GetAppLogger() const;

 private:
  const OrtLoggingLevel severity_;
  const std::string logger_id_;
  const spdlog::sink_ptr sink_;
  const std::shared_ptr<spdlog::logger> default_logger_;

  Ort::Env runtime_environment_;
  Ort::SessionOptions options_;
  Ort::Session session;
  std::vector<std::string> model_output_names_;
};

}  // namespace server
}  // namespace onnxruntime
