// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "core/common/logging/logging.h"
#include "environment.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "logging/syslog_sink.h"

namespace onnxruntime {
namespace server {

void ORT_API_CALL Log(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message){
      
      return;
    }

ServerEnvironment::ServerEnvironment(logging::Severity severity, logging::LoggingManager::InstanceType instance_type) : severity_(severity),
                                                                     logger_id_("ServerApp"),
                                                                     default_logging_manager_(
                                                                         std::unique_ptr<logging::ISink>{new SysLogSink{}},
                                                                         severity,
                                                                         /* default_filter_user_data */ false,
                                                                         instance_type,
                                                                         &logger_id_),
                                                                     runtime_environment_((OrtLoggingLevel) severity, Log, "ServerApp", nullptr), 
                                                                     session(nullptr) {
}

common::Status ServerEnvironment::InitializeModel(const std::string& model_path) {
  session = Ort::Session(runtime_environment_, model_path.c_str(), Ort::SessionOptions());

  auto outputCount = session.GetOutputCount();
  
  auto allocator = Ort::Allocator::CreateDefault();
  for (size_t i = 0; i<outputCount; i++) {
    auto name = session.GetOutputName(i, allocator);
    model_output_names_.push_back(name);
    allocator.Free(name);
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

const Ort::Session& ServerEnvironment::GetSession() const {
  return session;
}

}  // namespace server
}  // namespace onnxruntime
