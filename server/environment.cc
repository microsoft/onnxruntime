// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "environment.h"
#include "onnxruntime_cxx_api.h"

#ifdef USE_DNNL

#include "core/providers/dnnl/dnnl_provider_factory.h"

#endif

#ifdef USE_NUPHAR

#include "core/providers/nuphar/nuphar_provider_factory.h"

#endif


namespace onnxruntime {
namespace server {

static spdlog::level::level_enum Convert(OrtLoggingLevel in) {
  switch (in) {
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE:
      return spdlog::level::level_enum::debug;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO:
      return spdlog::level::level_enum::info;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING:
      return spdlog::level::level_enum::warn;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR:
      return spdlog::level::level_enum::err;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL:
      return spdlog::level::level_enum::critical;
    default:
      return spdlog::level::level_enum::off;
  }
}

void ORT_API_CALL Log(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
                      const char* message) {
  spdlog::logger* logger = static_cast<spdlog::logger*>(param);
  logger->log(Convert(severity), "[{} {} {}]: {}", logid, category, code_location, message);
  return;
}

ServerEnvironment::ServerEnvironment(OrtLoggingLevel severity, spdlog::sinks_init_list sink) : severity_(severity),
                                                                                               logger_id_("ServerApp"),
                                                                                               sink_(sink),
                                                                                               default_logger_(std::make_shared<spdlog::logger>(logger_id_, sink)),
                                                                                               runtime_environment_(severity, logger_id_.c_str(), Log, default_logger_.get()) {
  spdlog::set_automatic_registration(false);
  spdlog::set_level(Convert(severity_));
  spdlog::initialize_logger(default_logger_);
}

void ServerEnvironment::RegisterExecutionProviders(){
  #ifdef USE_DNNL
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(options_, 1));
  #endif

  #ifdef USE_NUPHAR
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(options_, 1, ""));
  #endif
}

void ServerEnvironment::InitializeModel(const std::string& model_path, const std::string& model_name, const std::string& model_version) {
  RegisterExecutionProviders();
  auto result = sessions_.emplace(std::piecewise_construct, std::forward_as_tuple(model_name, model_version), std::forward_as_tuple(runtime_environment_, model_path.c_str(), options_));

  if (!result.second) {
    throw Ort::Exception("Model of that name already loaded.", ORT_INVALID_ARGUMENT);
  }

  auto iterator = result.first;
  auto output_count = (iterator->second).session.GetOutputCount();

  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < output_count; i++) {
    auto name = (iterator->second).session.GetOutputName(i, allocator);
    (iterator->second).output_names.push_back(name);
    allocator.Free(name);
  }
}

const std::vector<std::string>& ServerEnvironment::GetModelOutputNames(const std::string& model_name, const std::string& model_version) const {
  auto identifier = std::make_pair(model_name, model_version);
  auto it = sessions_.find(identifier);
  if (it == sessions_.end()) {
    throw Ort::Exception("No model loaded of that name.", ORT_NO_MODEL);
  }

  return it->second.output_names;
}

OrtLoggingLevel ServerEnvironment::GetLogSeverity() const {
  return severity_;
}

const Ort::Session& ServerEnvironment::GetSession(const std::string& model_name, const std::string& model_version) const {
  auto identifier = std::make_pair(model_name, model_version);
  auto it = sessions_.find(identifier);
  if (it == sessions_.end()) {
    throw Ort::Exception("No model loaded of that name.", ORT_NO_MODEL);
  }

  return it->second.session;
}

std::shared_ptr<spdlog::logger> ServerEnvironment::GetLogger(const std::string& request_id) const {
  auto logger = std::make_shared<spdlog::logger>(request_id, sink_.begin(), sink_.end());
  spdlog::initialize_logger(logger);
  return logger;
}

std::shared_ptr<spdlog::logger> ServerEnvironment::GetAppLogger() const {
  return default_logger_;
}

void ServerEnvironment::UnloadModel(const std::string& model_name, const std::string& model_version) {
  auto identifier = std::make_pair(model_name, model_version);
  auto it = sessions_.find(identifier);
  if (it == sessions_.end()) {
    throw Ort::Exception("No model loaded of that name.", ORT_NO_MODEL);
  }

  sessions_.erase(it);
}

}  // namespace server
}  // namespace onnxruntime
