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
#include <unordered_map>
#include <vector>

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"
#include <mutex>

namespace onnxruntime {
namespace logging {

class EtwSink : public ISink {
 public:
  EtwSink() : ISink(SinkType::EtwSink) {}
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
  enum class InitializationStatus { NotInitialized,
                                    Initializing,
                                    Initialized,
                                    Failed };

 public:
  using EtwInternalCallback = std::function<void(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level,
                                                 ULONGLONG MatchAnyKeyword, ULONGLONG MatchAllKeyword,
                                                 PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext)>;

  // Singleton instance access
  static EtwRegistrationManager& Instance();

  // Returns true if ETW is supported at all.
  static bool SupportsETW();

  // Check if ETW logging is enabled
  bool IsEnabled() const;

  // Get the current logging level
  UCHAR Level() const;

  Severity MapLevelToSeverity();

  // Get the current keyword
  uint64_t Keyword() const;

  // Get the ETW registration status
  HRESULT Status() const;

  void RegisterInternalCallback(const std::string& cb_key, EtwInternalCallback callback);

  void UnregisterInternalCallback(const std::string& cb_key);

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

  std::mutex init_mutex_;
  std::atomic<InitializationStatus> initialization_status_ = InitializationStatus::NotInitialized;
  std::unordered_map<std::string, EtwInternalCallback> callbacks_;
  std::mutex callbacks_mutex_;
  mutable std::mutex provider_change_mutex_;
  bool is_enabled_;
  UCHAR level_;
  ULONGLONG keyword_;
  HRESULT etw_status_;
};

}  // namespace logging
}  // namespace onnxruntime
#else
// ETW is not supported on this platform but should still define a dummy EtwRegistrationManager
// so that it can be used in the EP provider bridge.
#include "core/common/logging/severity.h"

namespace onnxruntime {
namespace logging {
class EtwRegistrationManager {
 public:
  using EtwInternalCallback = std::function<void(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level,
                                                 ULONGLONG MatchAnyKeyword, ULONGLONG MatchAllKeyword,
                                                 PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext)>;

  static EtwRegistrationManager& Instance();
  static bool SupportsETW();
  bool IsEnabled() const { return false; }
  UCHAR Level() const { return 0; }
  Severity MapLevelToSeverity() { return Severity::kFATAL; }
  uint64_t Keyword() const { return 0; }
  HRESULT Status() const { return 0; }
  void RegisterInternalCallback(const std::string& cb_key, EtwInternalCallback callback) {}
  void UnregisterInternalCallback(const std::string& cb_key) {}

 private:
  EtwRegistrationManager() = default;
  ~EtwRegistrationManager() = default;
};
}  // namespace logging
}  // namespace onnxruntime
#endif  // ETW_TRACE_LOGGING_SUPPORTED
