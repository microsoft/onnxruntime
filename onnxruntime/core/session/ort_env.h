// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <string>
#include "core/session/onnxruntime_c_api.h"
#include "core/common/logging/isink.h"
#include "core/platform/ort_mutex.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
class Environment;
}

class LoggingWrapper : public onnxruntime::logging::ISink {
 public:
  LoggingWrapper(OrtLoggingFunction logging_function, void* logger_param);

  void SendImpl(const onnxruntime::logging::Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id,
                const onnxruntime::logging::Capture& message) override;

 private:
  OrtLoggingFunction logging_function_;
  void* logger_param_;
};

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

  static OrtEnv* GetInstance(const LoggingManagerConstructionInfo& lm_info,
                             onnxruntime::common::Status& status,
                             const OrtThreadingOptions* tp_options = nullptr);

  static void Release(OrtEnv* env_ptr);

  const onnxruntime::Environment& GetEnvironment() const {
    return *(value_.get());
  }

  onnxruntime::logging::LoggingManager* GetLoggingManager() const;
  void SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager);

  /**
   * Registers an allocator for sharing between multiple sessions.
   * Returns an error if an allocator with the same OrtMemoryInfo is already registered.
  */
  onnxruntime::Status RegisterAllocator(onnxruntime::AllocatorPtr allocator);

 private:
  static OrtEnv* p_instance_;
  static onnxruntime::OrtMutex m_;
  static int ref_count_;

  std::unique_ptr<onnxruntime::Environment> value_;

  OrtEnv(std::unique_ptr<onnxruntime::Environment> value1);
  ~OrtEnv() = default;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtEnv);
};