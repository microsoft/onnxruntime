// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <memory>
#include "core/platform/ort_mutex.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
/**
   Set up the runtime environment for onnxruntime.
   Note that in the current implementation environment can only be created once per process.
*/
class Environment {
 public:

  Environment(std::unique_ptr<logging::LoggingManager> loggingManager)
    : loggingManager_(std::move(loggingManager)) {
  }
    
  /**
     Create and initialize the runtime environment.
  */
  Status Initialize(const std::string& default_logger_id);

  /**
     Return if a runtime environment instance has been created and initialized.
  */
  static bool IsInitialized();

  /**
     Return the default logger set up as part of the environment.
  */
  static const logging::Logger& DefaultLogger();
 
  /**
     Logs a FATAL level message and creates an exception that can be thrown with error information.
     @param category The log category.
     @param location The location the log message was generated.
     @param format_str The printf format string.
     @param ... The printf arguments.
     @returns A new Logger instance that the caller owns.
  */
  std::exception LogFatalAndCreateException(const char* category,
                                            const CodeLocation& location,
                                            const char* format_str, ...);

  /**
     This function will call ::google::protobuf::ShutdownProtobufLibrary
  */
  ~Environment();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);

  std::unique_ptr<logging::LoggingManager> loggingManager_;
  static OrtMutex s_env_mutex_;
  static logging::Logger* s_default_logger_;
  static bool s_env_initialized_;

  Environment() = default;
};

inline bool Environment::IsInitialized() {
  std::lock_guard<OrtMutex> guard(s_env_mutex_);
  return s_env_initialized_;
}

inline const logging::Logger& Environment::DefaultLogger() {
  if (s_default_logger_ == nullptr) {
    // fail early for attempted misuse. don't use logging macros as we have no logger.
    throw std::logic_error("Attempt to use DefaultLogger but none has been registered.");
  }
  return *s_default_logger_;
}

}  // namespace onnxruntime
