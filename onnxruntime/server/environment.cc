// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "core/common/logging/logging.h"

#include "environment.h"
#include "log_sink.h"

namespace onnxruntime {
namespace server {

ServerEnvironment::ServerEnvironment(logging::Severity severity, logging::LoggingManager::InstanceType instance_type, bool env_init) : severity_(severity),
                                                                     logger_id_("ServerApp"),
                                                                     default_logging_manager_(
                                                                         std::unique_ptr<logging::ISink>{new LogSink{}},
                                                                         severity,
                                                                         /* default_filter_user_data */ false,
                                                                         instance_type,
                                                                         &logger_id_) {
  if (env_init) {
    auto status = onnxruntime::Environment::Create(runtime_environment_);
  }

  // The session initialization MUST BE AFTER environment creation
  session = std::make_unique<onnxruntime::InferenceSession>(options_, &default_logging_manager_);
}

common::Status ServerEnvironment::InitializeModel(const std::string& model_path) {
  auto status = session->Load(model_path);
  if (!status.IsOK()) {
    return status;
  }

  auto outputs = session->GetModelOutputs();
  if (!outputs.first.IsOK()) {
    return outputs.first;
  }

  for (const auto* output_node : *(outputs.second)) {
    model_output_names_.push_back(output_node->Name());
  }

  return common::Status::OK();
}

const std::vector<std::string>& ServerEnvironment::GetModelOutputNames() const {
  return model_output_names_;
}

const logging::Logger& ServerEnvironment::GetAppLogger() const {
  return default_logging_manager_.DefaultLogger();
}

logging::Severity ServerEnvironment::GetLogSeverity() const {
  return severity_;
}

std::unique_ptr<logging::Logger> ServerEnvironment::GetLogger(const std::string& id) {
  if (id.empty()) {
    LOGS(GetAppLogger(), WARNING) << "Request id is null or empty string";
  }

  return default_logging_manager_.CreateLogger(id, severity_, false);
}

onnxruntime::InferenceSession* ServerEnvironment::GetSession() const {
  return session.get();
}

}  // namespace server
}  // namespace onnxruntime
