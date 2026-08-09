// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _WIN32
#include <Windows.h>

#if !BUILD_QNN_EP_STATIC_LIB
#include <TraceLoggingProvider.h>
#endif

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"

#if !BUILD_QNN_EP_STATIC_LIB
TRACELOGGING_DECLARE_PROVIDER(telemetry_provider_handle);
#endif

namespace onnxruntime {
namespace qnn {

/// <summary>
/// Singleton class used to log QNN profiling events to the ONNX Runtime telemetry tracelogging provider.
///
/// When QNN EP is a DLL, we must define our own tracelogging provider handle via TRACELOGGING_DEFINE_PROVIDER.
/// TraceLogging documentation states that separate DLLs cannot share the same tracelogging provider handle. See:
/// https://learn.microsoft.com/en-us/windows/win32/api/traceloggingprovider/nf-traceloggingprovider-tracelogging_define_provider#remarks
///
/// When QNN EP is a static library, we use the tracelogging provider handle already defined
/// in core/platform/windows/telemetry.h/.cc. In this case, we forward method calls to the
/// ORT Env's telemetry provider.
/// </summary>
class QnnTelemetry {
 public:
  static QnnTelemetry& Instance();
  bool IsEnabled() const;

  // Get the current logging level
  unsigned char Level() const;

  // Get the current keyword
  UINT64 Keyword() const;

  // Logs QNN profiling event as trace logging event.
  void LogQnnProfileEvent(uint64_t timestamp,
                          const std::string& message,
                          const std::string& qnnScalarValue,
                          const std::string& unit,
                          const std::string& timingSource,
                          const std::string& eventLevel,
                          const char* eventIdentifier) const;

  using EtwInternalCallback = std::function<void(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level,
                                                 ULONGLONG MatchAnyKeyword, ULONGLONG MatchAllKeyword,
                                                 PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext)>;

  static void RegisterInternalCallback(const EtwInternalCallback& callback);

  static void UnregisterInternalCallback(const EtwInternalCallback& callback);

 private:
  QnnTelemetry();
  ~QnnTelemetry();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnTelemetry);

#if !BUILD_QNN_EP_STATIC_LIB
  static std::mutex mutex_;
  static uint32_t global_register_count_;
  static bool enabled_;

  static std::vector<const EtwInternalCallback*> callbacks_;
  static std::mutex callbacks_mutex_;
  static std::mutex provider_change_mutex_;
  static UCHAR level_;
  static ULONGLONG keyword_;

  static void InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                              ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext);

  static void NTAPI ORT_TL_EtwEnableCallback(
      _In_ LPCGUID SourceId,
      _In_ ULONG IsEnabled,
      _In_ UCHAR Level,
      _In_ ULONGLONG MatchAnyKeyword,
      _In_ ULONGLONG MatchAllKeyword,
      _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData,
      _In_opt_ PVOID CallbackContext);
#endif
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // defined(_WIN32)
