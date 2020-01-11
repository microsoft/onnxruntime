// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "core/common/logging/isink.h"
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include "WinMLAdapter.h"

#pragma warning(push)
#pragma warning(disable : 4505)

namespace Windows::AI ::MachineLearning {

class CWinMLLogSink : public onnxruntime::logging::ISink {
 public:
  CWinMLLogSink() {
  }
  static void EnableDebugOutput() {
    debug_output_ = true;
    OutputDebugStringW(L"Windows.AI.MachineLearning: Debug Output Enabled \r\n");
  }
  void SendProfileEvent(onnxruntime::profiling::EventRecord& event_record) const;
  void SendImpl(const onnxruntime::logging::Timestamp& timestamp, const std::string& logger_id, const onnxruntime::logging::Capture& message);

 private:
  static bool debug_output_;
};
// TODO: a bug in ORT requires a logging manager.  This function registers a static singleton logger as "default"
inline onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
  // create a CLog based default logging manager
  static std::string default_logger_id{"Default"};
  static onnxruntime::logging::LoggingManager default_logging_manager{
      std::unique_ptr<onnxruntime::logging::ISink>{new CWinMLLogSink()},
      onnxruntime::logging::Severity::kVERBOSE,
      false,
      onnxruntime::logging::LoggingManager::InstanceType::Default,
      &default_logger_id,
      MAXINT32};

  return default_logging_manager;
}

class LotusEnvironment {
 public:
  LotusEnvironment() {
    // TODO: Do we need to call this or just define the method?
    default_logging_manager_ = &DefaultLoggingManager();

  }


  const onnxruntime::logging::Logger* GetDefaultLogger() {
    return &default_logging_manager_->DefaultLogger();
  }

 private:
  onnxruntime::logging::LoggingManager* default_logging_manager_;
};

}  // namespace MachineLearning::AI::Windows

#pragma warning(pop)