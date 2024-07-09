// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <Windows.h>
#include <ntverp.h>
#include <evntrace.h>

// check for Windows 10 SDK or later
// https://stackoverflow.com/questions/2665755/how-can-i-determine-the-version-of-the-windows-sdk-installed-on-my-computer
#if VER_PRODUCTBUILD > 9600
// ETW trace logging uses Windows 10 SDK's TraceLoggingProvider.h
#define ETW_TRACE_LOGGING_SUPPORTED 1
#endif

#ifdef ETW_TRACE_LOGGING_SUPPORTED

#include <date/date.h>
#include <atomic>
#include <iostream>
#include <string>
#include <vector>

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace logging {

class EtwSink : public ISink {
 public:
  EtwSink() = default;
  ~EtwSink() = default;

  constexpr static const char* kEventName = "ONNXRuntimeLogEvent";

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EtwSink);

  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override;

  // limit to one instance of an EtwSink being around, so we can control the lifetime of
  // EtwTracingManager to ensure we cleanly unregister it
  static std::atomic_flag have_instance_;
};

class EtwRegistrationManager {
 public:
  using EtwInternalCallback = std::function<void(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level,
                                                 ULONGLONG MatchAnyKeyword, ULONGLONG MatchAllKeyword,
                                                 PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext)>;

  // Singleton instance access
  static EtwRegistrationManager& Instance();

  // Check if ETW logging is enabled
  bool IsEnabled() const;

  // Get the current logging level
  UCHAR Level() const;

  Severity MapLevelToSeverity();

  // Get the current keyword
  uint64_t Keyword() const;

  void RegisterInternalCallback(const EtwInternalCallback& callback);

 private:
  EtwRegistrationManager();
  ~EtwRegistrationManager();
  void LazyInitialize();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EtwRegistrationManager);

  void InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                       ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext);

  static void NTAPI ORT_TL_EtwEnableCallback(
      _In_ LPCGUID SourceId,
      _In_ ULONG IsEnabled,
      _In_ UCHAR Level,
      _In_ ULONGLONG MatchAnyKeyword,
      _In_ ULONGLONG MatchAllKeyword,
      _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData,
      _In_opt_ PVOID CallbackContext);

  std::vector<EtwInternalCallback> callbacks_;
  OrtMutex callbacks_mutex_;
  mutable OrtMutex provider_change_mutex_;
  OrtMutex init_mutex_;
  bool initialized_ = false;
  bool is_enabled_;
  UCHAR level_;
  ULONGLONG keyword_;
};

}  // namespace logging
}  // namespace onnxruntime

#endif  // ETW_TRACE_LOGGING_SUPPORTED
