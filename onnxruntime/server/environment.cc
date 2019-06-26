// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "environment.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace server {

void ORT_API_CALL Log(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
                      const char* message) {
  spdlog::logger* logger = (spdlog::logger*)param;
  logger->log((spdlog::level::level_enum)(severity - 1), "[{} {} {}]: {}", logid, category, code_location, message);
  return;
}

ServerEnvironment::ServerEnvironment(OrtLoggingLevel severity, spdlog::sink_ptr sink) : severity_(severity),
                                                                                        logger_id_("ServerApp"),
                                                                                        sink_(sink),
                                                                                        default_logger_(std::make_shared<spdlog::logger>(logger_id_, sink_)),
                                                                                        runtime_environment_((OrtLoggingLevel)severity, logger_id_.c_str(), Log, default_logger_.get()),
                                                                                        session(nullptr) {
  spdlog::set_automatic_registration(false);
  spdlog::set_level((spdlog::level::level_enum)severity_);
  spdlog::initialize_logger(default_logger_);
}

void ServerEnvironment::InitializeModel(const std::string& model_path) {
  session = Ort::Session(runtime_environment_, model_path.c_str(), Ort::SessionOptions());

  auto outputCount = session.GetOutputCount();

  auto allocator = Ort::Allocator::CreateDefault();
  for (size_t i = 0; i < outputCount; i++) {
    auto name = session.GetOutputName(i, allocator);
    model_output_names_.push_back(name);
    allocator.Free(name);
  }
}

const std::vector<std::string>& ServerEnvironment::GetModelOutputNames() const {
  return model_output_names_;
}

OrtLoggingLevel ServerEnvironment::GetLogSeverity() const {
  return severity_;
}

const Ort::Session& ServerEnvironment::GetSession() const {
  return session;
}

std::shared_ptr<spdlog::logger> ServerEnvironment::GetLogger(const std::string& request_id) const {
  auto logger = std::make_shared<spdlog::logger>(request_id, sink_);
  spdlog::initialize_logger(logger);
  return logger;
}

std::shared_ptr<spdlog::logger> ServerEnvironment::GetAppLogger() const {
  return default_logger_;
}

}  // namespace server
}  // namespace onnxruntime
