// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "core/common/logging/logging.h"

#include "environment.h"
#include "log_sink.h"

namespace onnxruntime {
namespace hosting {

HostingEnvironment::HostingEnvironment(onnxruntime::logging::Severity severity) : logger_id_("HostingApp"),
                                                                                  default_logging_manager_(
                                                                                      std::unique_ptr<onnxruntime::logging::ISink>{&sink_},
                                                                                      severity,
                                                                                      /* default_filter_user_data */ false,
                                                                                      onnxruntime::logging::LoggingManager::InstanceType::Default,
                                                                                      &logger_id_) {
  auto status = onnxruntime::Environment::Create(this->runtime_environment_);

  // The session initialization MUST BE AFTER environment creation
  session_ = std::make_shared<onnxruntime::InferenceSession>(options_, &default_logging_manager_);
}

const onnxruntime::logging::Logger& HostingEnvironment::GetAppLogger() {
  return this->default_logging_manager_.DefaultLogger();
}

std::shared_ptr<onnxruntime::logging::Logger> HostingEnvironment::GetLogger(const std::string& id) {
  if (id.empty()) {
    LOGS(GetAppLogger(), WARNING) << "Request id is null or empty string";
  }

  return this->default_logging_manager_.CreateLogger(id);
}

std::shared_ptr<onnxruntime::InferenceSession> HostingEnvironment::GetSession() const {
  return this->session_;
}

}  // namespace hosting
}  // namespace onnxruntime
