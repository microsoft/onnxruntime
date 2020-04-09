// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//this file contains implementations of the C API

#include <cassert>

#include "ort_env.h"
#include "core/session/ort_apis.h"
#include "core/session/environment.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/logging.h"

using namespace onnxruntime;
using namespace onnxruntime::logging;

OrtEnv* OrtEnv::p_instance_ = nullptr;
int OrtEnv::ref_count_ = 0;
onnxruntime::OrtMutex OrtEnv::m_;

LoggingWrapper::LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param)
    : logging_function_(logging_function), logger_param_(logger_param) {
}

void LoggingWrapper::SendImpl(const onnxruntime::logging::Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id,
                              const onnxruntime::logging::Capture& message) {
  std::string s = message.Location().ToString();
  logging_function_(logger_param_, static_cast<OrtLoggingLevel>(message.Severity()), message.Category(),
                    logger_id.c_str(), s.c_str(), message.Message().c_str());
}

OrtEnv::OrtEnv(std::unique_ptr<onnxruntime::Environment> value1)
    : value_(std::move(value1)) {
}

OrtEnv* OrtEnv::GetInstance(const OrtEnv::LoggingManagerConstructionInfo& lm_info,
                            onnxruntime::common::Status& status,
                            const OrtThreadingOptions* tp_options) {
  std::lock_guard<onnxruntime::OrtMutex> lock(m_);
  std::unique_ptr<LoggingManager> lmgr;
  std::string name = lm_info.logid;
  if (lm_info.logging_function) {
    std::unique_ptr<ISink> logger = onnxruntime::make_unique<LoggingWrapper>(lm_info.logging_function,
                                                                             lm_info.logger_param);
    lmgr.reset(new LoggingManager(std::move(logger),
                                  static_cast<Severity>(lm_info.default_warning_level),
                                  false,
                                  LoggingManager::InstanceType::Default,
                                  &name));
  } else {
    lmgr.reset(new LoggingManager(std::unique_ptr<ISink>{new CLogSink{}},
                                  static_cast<Severity>(lm_info.default_warning_level),
                                  false,
                                  LoggingManager::InstanceType::Default,
                                  &name));
  }

  if (!p_instance_) {
    std::unique_ptr<onnxruntime::Environment> env;
    if (!tp_options) {
      status = onnxruntime::Environment::Create(std::move(lmgr), env);
    } else {
      status = onnxruntime::Environment::Create(std::move(lmgr), env, tp_options, true);
    }
    if (!status.IsOK()) {
      return nullptr;
    }
    p_instance_ = new OrtEnv(std::move(env));
  }
  ++ref_count_;
  return p_instance_;
}

void OrtEnv::Release(OrtEnv* env_ptr) {
  if (!env_ptr) {
    return;
  }
  std::lock_guard<onnxruntime::OrtMutex> lock(m_);
  ORT_ENFORCE(env_ptr == p_instance_);  // sanity check
  --ref_count_;
  if (ref_count_ == 0) {
    delete p_instance_;
    p_instance_ = nullptr;
  }
}

onnxruntime::logging::LoggingManager* OrtEnv::GetLoggingManager() const {
  return value_->GetLoggingManager();
}

void OrtEnv::SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager) {
  value_->SetLoggingManager(std::move(logging_manager));
}