// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_telemetry.h"

#ifdef _WIN32
#if !BUILD_QNN_EP_STATIC_LIB
// ETW includes
// need space after Windows.h to prevent clang-format re-ordering breaking the build.
// TraceLoggingProvider.h must follow Windows.h
#include <Windows.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26440)  // Warning C26440 from TRACELOGGING_DEFINE_PROVIDER
#endif

#include <TraceLoggingProvider.h>
#include <evntrace.h>
#include <winmeta.h>
#include "core/platform/windows/TraceLoggingConfig.h"

// Seems this workaround can be dropped when we drop support for VS2017 toolchains
// https://developercommunity.visualstudio.com/content/problem/85934/traceloggingproviderh-is-incompatible-with-utf-8.html
#ifdef _TlgPragmaUtf8Begin
#undef _TlgPragmaUtf8Begin
#define _TlgPragmaUtf8Begin
#endif

#ifdef _TlgPragmaUtf8End
#undef _TlgPragmaUtf8End
#define _TlgPragmaUtf8End
#endif

// Different versions of TraceLoggingProvider.h contain different macro variable names for the utf8 begin and end,
// and we need to cover the lower case version as well.
#ifdef _tlgPragmaUtf8Begin
#undef _tlgPragmaUtf8Begin
#define _tlgPragmaUtf8Begin
#endif

#ifdef _tlgPragmaUtf8End
#undef _tlgPragmaUtf8End
#define _tlgPragmaUtf8End
#endif

TRACELOGGING_DEFINE_PROVIDER(telemetry_provider_handle, "Microsoft.ML.ONNXRuntime",
                             // {3a26b1ff-7484-7484-7484-15261f42614d}
                             (0x3a26b1ff, 0x7484, 0x7484, 0x74, 0x84, 0x15, 0x26, 0x1f, 0x42, 0x61, 0x4d),
                             TraceLoggingOptionMicrosoftTelemetry());

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif  // !BUILD_QNN_EP_STATIC_LIB

#include "core/providers/qnn/ort_api.h"
#include <unordered_map>

namespace onnxruntime {
namespace qnn {

#if !BUILD_QNN_EP_STATIC_LIB
std::mutex QnnTelemetry::mutex_;
std::mutex QnnTelemetry::provider_change_mutex_;
uint32_t QnnTelemetry::global_register_count_ = 0;
bool QnnTelemetry::enabled_ = true;
UCHAR QnnTelemetry::level_ = 0;
UINT64 QnnTelemetry::keyword_ = 0;
std::unordered_map<std::string, QnnTelemetry::EtwInternalCallback> QnnTelemetry::callbacks_;
std::mutex QnnTelemetry::callbacks_mutex_;
#endif  // !BUILD_QNN_EP_STATIC_LIB

QnnTelemetry::QnnTelemetry() {
#if !BUILD_QNN_EP_STATIC_LIB
  std::lock_guard<std::mutex> lock(mutex_);
  if (global_register_count_ == 0) {
    // TraceLoggingRegister is fancy in that you can only register once GLOBALLY for the whole process
    HRESULT hr = TraceLoggingRegisterEx(telemetry_provider_handle, ORT_TL_EtwEnableCallback, nullptr);
    if (SUCCEEDED(hr)) {
      global_register_count_ += 1;
    }
  }
#endif  // !BUILD_QNN_EP_STATIC_LIB
}

QnnTelemetry::~QnnTelemetry() {
#if !BUILD_QNN_EP_STATIC_LIB
  std::lock_guard<std::mutex> lock(mutex_);
  if (global_register_count_ > 0) {
    global_register_count_ -= 1;
    if (global_register_count_ == 0) {
      TraceLoggingUnregister(telemetry_provider_handle);
    }
  }

  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.clear();
#endif  // !BUILD_QNN_EP_STATIC_LIB
}

QnnTelemetry& QnnTelemetry::Instance() {
  static QnnTelemetry instance;
  return instance;
}

bool QnnTelemetry::IsEnabled() const {
#if BUILD_QNN_EP_STATIC_LIB
  const Env& env = GetDefaultEnv();
  auto& provider = env.GetTelemetryProvider();
  return provider.IsEnabled();
#else
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return enabled_;
#endif
}

UCHAR QnnTelemetry::Level() const {
#if BUILD_QNN_EP_STATIC_LIB
  const Env& env = GetDefaultEnv();
  auto& provider = env.GetTelemetryProvider();
  return provider.Level();
#else
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return level_;
#endif
}

UINT64 QnnTelemetry::Keyword() const {
#if BUILD_QNN_EP_STATIC_LIB
  const Env& env = GetDefaultEnv();
  auto& provider = env.GetTelemetryProvider();
  return provider.Keyword();
#else
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return keyword_;
#endif
}

void QnnTelemetry::LogQnnProfileEvent(uint64_t timestamp,
                                      const std::string& message,
                                      const std::string& qnnScalarValue,
                                      const std::string& unit,
                                      const std::string& timingSource,
                                      const std::string& eventLevel,
                                      const char* eventIdentifier) const {
  TraceLoggingWrite(
      telemetry_provider_handle,
      "QNNProfilingEvent",
      TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)),
      TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
      TraceLoggingValue(timestamp, "Timestamp"),
      TraceLoggingString(message.c_str(), "Message"),
      TraceLoggingString(qnnScalarValue.c_str(), "Value"),
      TraceLoggingString(unit.c_str(), "Unit of Measurement"),
      TraceLoggingString(timingSource.c_str(), "Timing Source"),
      TraceLoggingString(eventLevel.c_str(), "Event Level"),
      TraceLoggingString(eventIdentifier, "Event Identifier"));
}

void QnnTelemetry::RegisterInternalCallback(const std::string& cb_key, EtwInternalCallback callback) {
#if BUILD_QNN_EP_STATIC_LIB
  WindowsTelemetry::RegisterInternalCallback(cb_key, std::move(callback));
#else
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.insert_or_assign(cb_key, std::move(callback));
#endif
}

void QnnTelemetry::UnregisterInternalCallback(const std::string& cb_key) {
#if BUILD_QNN_EP_STATIC_LIB
  WindowsTelemetry::UnregisterInternalCallback(cb_key);
#else
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.erase(cb_key);
#endif
}

#if !BUILD_QNN_EP_STATIC_LIB
void NTAPI QnnTelemetry::ORT_TL_EtwEnableCallback(
    _In_ LPCGUID SourceId,
    _In_ ULONG IsEnabled,
    _In_ UCHAR Level,
    _In_ ULONGLONG MatchAnyKeyword,
    _In_ ULONGLONG MatchAllKeyword,
    _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData,
    _In_opt_ PVOID CallbackContext) {
  {
    std::lock_guard<std::mutex> lock(provider_change_mutex_);
    enabled_ = (IsEnabled != 0);
    level_ = Level;
    keyword_ = MatchAnyKeyword;
  }

  InvokeCallbacks(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
}

void QnnTelemetry::InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                                   ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData,
                                   PVOID CallbackContext) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  for (const auto& entry : callbacks_) {
    const auto& cb = entry.second;
    cb(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
  }
}
#endif  // !BUILD_QNN_EP_STATIC_LIB

}  // namespace qnn
}  // namespace onnxruntime
#endif  // defined(_WIN32)
