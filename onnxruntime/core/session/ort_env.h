// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <string>
#include <mutex>
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class Environment;
}

// Managed pointer type for OrtEnv that calls OrtEnv::Release as its deleter.
using OrtEnvPtr = std::unique_ptr<OrtEnv, void (*)(OrtEnv*)>;

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

  /// <summary>
  /// Gets or creates the global OrtEnv instance. Arguments are ignored if the instance has already been created.
  /// </summary>
  /// <param name="lm_info">Configuration for the logging manager.</param>
  /// <param name="status">Output parameter that indicates if an error occurred during environment creation.</param>
  /// <param name="tp_options">Optional threading options.</param>
  /// <param name="config_entries">Optional configuration entries.</param>
  /// <returns>The OrtEnv instance.</returns>
  static OrtEnvPtr GetOrCreateInstance(const LoggingManagerConstructionInfo& lm_info,
                                       onnxruntime::common::Status& status,
                                       const OrtThreadingOptions* tp_options = nullptr,
                                       const OrtKeyValuePairs* config_entries = nullptr);

  /// <summary>
  /// Gets the global OrtEnv instance. Returns nullptr if the instance has not yet been created.
  /// </summary>
  /// <returns>The OrtEnv instance or nullptr.</returns>
  static OrtEnvPtr TryGetInstance();

  static void Release(OrtEnv* env_ptr);

  const onnxruntime::Environment& GetEnvironment() const {
    return *value_;
  }

  onnxruntime::Environment& GetEnvironment() {
    return *value_;
  }

  onnxruntime::logging::LoggingManager* GetLoggingManager() const;
  void SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager);

  OrtEnv(std::unique_ptr<onnxruntime::Environment> value);
  ~OrtEnv();

 private:
  // p_instance_ holds the single, global instance of OrtEnv.
  // This is a raw pointer to allow for intentional memory leaking when
  // the process is shutting down (g_is_shutting_down is true).
  // Using a smart pointer like std::unique_ptr would complicate this specific
  // shutdown scenario, as it would attempt to deallocate the memory even if
  // Release() hasn't been called or if a leak is desired.
  // Management is handled by GetOrCreateInstance(), TryGetInstance(), and Release(), with ref_count_
  // tracking active users. It is set to nullptr when the last reference is released
  // (and not shutting down).
  static OrtEnv* p_instance_;
  static std::mutex m_;
  static int ref_count_;

  std::unique_ptr<onnxruntime::Environment> value_;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtEnv);
};
