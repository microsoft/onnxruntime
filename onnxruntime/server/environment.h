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
  explicit ServerEnvironment(OrtLoggingLevel severity, spdlog::sinks_init_list sink);
  ~ServerEnvironment() = default;
  ServerEnvironment(const ServerEnvironment&) = delete;

  OrtLoggingLevel GetLogSeverity() const;

  const Ort::Session& GetSession(const std::string& model_name, const std::string& model_version) const;
  void InitializeModel(const std::string& model_path, const std::string& model_name, const std::string& model_version);
  const std::vector<std::string>& GetModelOutputNames(const std::string& model_name, const std::string& model_version) const;
  std::shared_ptr<spdlog::logger> GetLogger(const std::string& request_id) const;
  std::shared_ptr<spdlog::logger> GetAppLogger() const;
  void UnloadModel(const std::string& model_name, const std::string& model_version);

 private:
  const OrtLoggingLevel severity_;
  const std::string logger_id_;
  const std::vector<spdlog::sink_ptr> sink_;
  const std::shared_ptr<spdlog::logger> default_logger_;

  Ort::Env runtime_environment_;
  Ort::SessionOptions options_;

  struct SesssionHolder {
      Ort::Session session;
      std::vector<std::string> output_names;
      explicit SessionHolder(Ort::Session in_session) : session(in_session)
      {
      };
      ~SessionHolder() = default;
      SessionHolder(const SessionHolder&) = delete;
      SessionHolder(const SessionHolder&&) = delete;
      SessionHolder& operator=(const SessionHolder&) = delete; 
  };

  std::map<std::string, SessionHolder> sessions_;
};

}  // namespace server
}  // namespace onnxruntime
