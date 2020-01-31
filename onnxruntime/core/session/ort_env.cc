// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_env.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
using namespace onnxruntime::logging;
using namespace onnxruntime;

class LoggingWrapper : public ISink {
 public:
  LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param)
      : logging_function_(logging_function), logger_param_(logger_param) {
  }

  void SendImpl(const Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id,
                const Capture& message) override {
    std::string s = message.Location().ToString();
    logging_function_(logger_param_, static_cast<OrtLoggingLevel>(message.Severity()), message.Category(),
                      logger_id.c_str(), s.c_str(), message.Message().c_str());
  }

 private:
  OrtLoggingFunction logging_function_;
  void* logger_param_;
};

OrtEnv* OrtEnv::GetInstance(const LoggingManagerConstructionInfo& lm_info, Status& status) {
  std::lock_guard<OrtMutex> lock(m_);
  if (!p_instance_) {
    std::unique_ptr<Environment> env;
    status = Environment::Create(env);
    if (!status.IsOK()) {
      return nullptr;
    }

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

    p_instance_ = new OrtEnv(std::move(env), std::move(lmgr));
  }
  ++ref_count_;
  return p_instance_;
}

OrtEnv* OrtEnv::p_instance_ = nullptr;
int OrtEnv::ref_count_ = 0;
OrtMutex OrtEnv::m_;