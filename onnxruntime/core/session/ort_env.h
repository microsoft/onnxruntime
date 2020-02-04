// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/status.h"
#include "core/platform/ort_mutex.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"

struct OrtEnv {
 public:
  struct LoggingManagerConstructionInfo {
    LoggingManagerConstructionInfo(OrtLoggingFunction logging_function1,
                                   void* logger_param1,
                                   OrtLoggingLevel default_warning_level1,
                                   const char* logid1)
        : logging_function(logging_function1),
          logger_param(logger_param1),
          default_warning_level(default_warning_level1),
          logid(logid1) {}
    OrtLoggingFunction logging_function{};
    void* logger_param{};
    OrtLoggingLevel default_warning_level;
    const char* logid{};
  };
  static OrtEnv* GetInstance(const LoggingManagerConstructionInfo& lm_info, onnxruntime::common::Status& status);

  static void Release(OrtEnv* env_ptr) {
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

  onnxruntime::logging::LoggingManager* GetLoggingManager() const {
    return logging_manager_.get();
  }

 private:
  static OrtEnv* p_instance_;
  static onnxruntime::OrtMutex m_;
  static int ref_count_;

  std::unique_ptr<onnxruntime::Environment> value_;
  std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager_;

  OrtEnv(std::unique_ptr<onnxruntime::Environment> value1, std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager)
      : value_(std::move(value1)), logging_manager_(std::move(logging_manager)) {
  }

  ~OrtEnv() = default;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtEnv);
};