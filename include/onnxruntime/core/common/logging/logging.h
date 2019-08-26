// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <chrono>
#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "core/common/common.h"
#include "core/common/logging/capture.h"
#include "core/common/logging/severity.h"

#include "core/common/logging/macros.h"

/*

  Logging overview and expected usage:

  At program startup:
  * Create one or more ISink instances. If multiple, combine using composite_sink.
  * Create a LoggingManager instance with the sink/s with is_default_instance set to true
  * Only one instance should be created in this way, and it should remain valid for
  until the program no longer needs to produce log output.

  You can either use the static default Logger which is created as part of the environment, 
  or separate Logger instances each with different log ids via LoggingManager::CreateLogger.

  The log id is passed to the ISink instance with the sink determining how the log id is used
  in the output.

  LoggingManager
  * creates the Logger instances used by the application
  * owns the log sink instance
  * applies checks on severity and output of user data

  The log macros create a Capture instance to capture the information to log.
  If the severity and/or user filtering settings would prevent logging, no evaluation
  of the log arguments will occur, so no performance cost beyond the severity and user
  filtering check.

  A sink can do further filter as needed.

*/

namespace onnxruntime {
namespace profiling {

enum EventCategory {
  SESSION_EVENT = 0,
  NODE_EVENT,
  EVENT_CATEGORY_MAX
};

/*
Event descriptions for the above session events.
*/
static constexpr const char* event_categor_names_[EVENT_CATEGORY_MAX] = {
    "Session",
    "Node"};

/*
Timing record for all events.
*/
struct EventRecord {
  EventRecord(EventCategory category, int process_id, int thread_id,
              std::string event_name, long long time_stamp, long long duration,
              std::unordered_map<std::string, std::string>&& event_args)
  : cat(category),
    pid(process_id),
    tid(thread_id),
    name(std::move(event_name)),
    ts(time_stamp),
    dur(duration),
    args(event_args) {}
  
  EventCategory cat;
  int pid;
  int tid;
  std::string name;
  long long ts;
  long long dur;
  std::unordered_map<std::string, std::string> args;
};
}  // namespace profiling

namespace logging {

using Timestamp = std::chrono::time_point<std::chrono::system_clock>;

#ifndef NDEBUG
ORT_ATTRIBUTE_UNUSED static bool vlog_enabled = true;  // Set directly based on your needs.
#else
constexpr bool vlog_enabled = false;  // no VLOG output
#endif

enum class DataType {
  SYSTEM = 0,  ///< System data.
  USER = 1     ///< Contains potentially sensitive user data.
};

// Internal log categories.
// Logging interface takes const char* so arbitrary values can also be used.
struct Category {
  static const char* onnxruntime;  ///< General output
  static const char* System;       ///< Log output regarding interactions with the host system
  // TODO: What other high level categories are meaningful? Model? Optimizer? Execution?
};

class ISink;
class Logger;
class Capture;

/// <summary>
/// The logging manager.
/// Provides filtering based on a minimum LogSeverity level, and of messages with DataType::User if enabled.
/// </summary>
class LoggingManager final {
 public:
  /**
     Initializes a new instance of the LoggingManager class.
     @param sink The sink to write to. Use CompositeSink if you need to write to multiple places.
     @param default_min_severity The default minimum severity. Messages with lower severity will be ignored unless
     overridden in CreateLogger.
     @param default_filter_user_data If set to true ignore messages with DataType::USER unless overridden in CreateLogger.
     @param default_logger_id Logger Id to use for the default logger. nullptr/ignored if instance_type == Temporal.
     @param default_max_vlog_level Default maximum level for VLOG messages to be created unless overridden in CreateLogger.
     Requires a severity of kVERBOSE for VLOG messages to be logged.
  */
  LoggingManager(std::unique_ptr<ISink> sink, Severity default_min_severity, bool default_filter_user_data,
                 int default_max_vlog_level = -1);

  ~LoggingManager();

  /**
     Creates a new logger instance which will use the provided logger_id and default severity and vlog levels.
     @param logger_id The log identifier.
     @returns A new Logger instance that the caller owns.
  */
  std::unique_ptr<Logger> CreateLogger(const std::string& logger_id);

  /**
     Creates a new logger instance which will use the provided logger_id, severity and vlog levels.
     @param logger_id The log identifier.
     @param min_severity The minimum severity. Requests to create messages with lower severity will be ignored.
     @param filter_user_data If set to true ignore messages with DataType::USER.
     @param max_vlog_level Maximum level for VLOG messages to be created.
     @returns A new Logger instance that the caller owns.
  */
  std::unique_ptr<Logger> CreateLogger(const std::string& logger_id, Severity min_severity,
                                       bool filter_user_data, int max_vlog_level = -1);

  /**
     Logs the message using the provided logger id.
     @param logger_id The log identifier.
     @param message The log message.
  */
  void Log(const std::string& logger_id, const Capture& message) const;

  /**
    Sends a Profiling Event Record to the sink.
    @param Profiling Event Record
  */
  void SendProfileEvent(profiling::EventRecord& eventRecord) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LoggingManager);

  Timestamp GetTimestamp() const noexcept;

  std::unique_ptr<ISink> sink_;
  const Severity default_min_severity_;
  const bool default_filter_user_data_;
  const int default_max_vlog_level_;

  struct Epochs {
    const std::chrono::time_point<std::chrono::high_resolution_clock> high_res;
    const std::chrono::time_point<std::chrono::system_clock> system;
    const std::chrono::minutes localtime_offset_from_utc;
  };

  static const Epochs& GetEpochs() noexcept;
};

/**
   Logger provides a per-instance log id. Everything else is passed back up to the LoggingManager
*/
class Logger {
 public:
  /**
     Initializes a new instance of the Logger class.
     @param loggingManager The logging manager.
     @param id The identifier for messages coming from this Logger.
     @param severity Minimum severity for messages to be created and logged.
     @param filter_user_data Should USER data be filtered from output.
     @param vlog_level Minimum level for VLOG messages to be created. Note that a severity of 
            kVERBOSE must be provided for VLOG messages to be logged.
  */
  Logger(const LoggingManager& loggingManager, std::string id,
         Severity severity, bool filter_user_data, int vlog_level)
      : logging_manager_{&loggingManager},
        id_{id},
        min_severity_{severity},
        filter_user_data_{filter_user_data},
        max_vlog_level_{severity > Severity::kVERBOSE ? -1 : vlog_level} {
  }

  /**
     Get the minimum severity level for log messages to be output.
     @returns The severity.
  */
  Severity GetSeverity() const noexcept { return min_severity_; }

  /**
     Change the minimum severity level for log messages to be output.
     @param severity The severity.
  */
  void SetSeverity(Severity severity) noexcept { min_severity_ = severity; }

  /**
     Check if output is enabled for the provided LogSeverity and DataType values.
     @param severity The severity.
     @param data_type Type of the data.
     @returns True if a message with these values will be logged.
  */
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept {
    return (severity >= min_severity_ && (data_type != DataType::USER || !filter_user_data_));
  }

  /**
     Return the maximum VLOG level allowed.
  */
  int VLOGMaxLevel() const noexcept {
    return max_vlog_level_;
  }

  /**
     Logs the captured message.
     @param message The log message.
  */
  void Log(const Capture& message) const {
    logging_manager_->Log(id_, message);
  }

  /**
    Sends a Profiling Event Record to the sink.
    @param Profiling Event Record
  */
  void SendProfileEvent(profiling::EventRecord& eventRecord) const {
    logging_manager_->SendProfileEvent(eventRecord);
  }

 private:
  const LoggingManager* logging_manager_;
  const std::string id_;
  Severity min_severity_;
  const bool filter_user_data_;
  const int max_vlog_level_;
};

inline Timestamp LoggingManager::GetTimestamp() const noexcept {
  static const Epochs& epochs = GetEpochs();

  const auto high_res_now = std::chrono::high_resolution_clock::now();
  return std::chrono::time_point_cast<std::chrono::system_clock::duration>(
      epochs.system + (high_res_now - epochs.high_res) + epochs.localtime_offset_from_utc);
}

/**
   Return the current thread id.
*/
unsigned int GetThreadId();

/**
   Return the current process id.
*/
unsigned int GetProcessId();

}  // namespace logging
}  // namespace onnxruntime
